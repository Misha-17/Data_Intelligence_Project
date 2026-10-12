"""Exercise the complete construction pipeline on small adversarial raw files."""
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from armd.build import KEY, LINK, build_dataset, profile_dataset
from armd.evaluate import _feature_frame, _numeric_features
from armd.support_features import NUMERIC_FEATURES


class ConstructionIntegrationTests(unittest.TestCase):
    def test_raw_sources_preserve_grain_and_exclude_bad_labels_and_future_history(self):
        root = Path(__file__).parents[1]
        config = json.loads((root / "config.json").read_text())
        config.update(sample_fraction=1., include_clinical_features=True)
        drugs = config["antibiotics"]
        self.assertEqual(len(drugs), 9)
        with tempfile.TemporaryDirectory() as directory:
            raw, output = Path(directory) / "raw", Path(directory) / "output"
            raw.mkdir()

            def write(suffix, rows):
                pd.DataFrame(rows).to_csv(raw / f"microbiology_cultures_{suffix}.csv", index=False)

            def record(patient, culture, drug, label="S", organism="E. coli", **extra):
                return {"anon_id": patient, "order_proc_id_coded": culture,
                        "order_time_jittered_utc": "2020-01-01 00:00:00+00:00",
                        "culture_description": "urine", "organism": organism,
                        "antibiotic": drug, "susceptibility": label,
                        "was_positive": "true", **extra}

            rows = [record("p1", "c1", drug) for drug in drugs]
            rows += [record("p1", "c1", drugs[0])]  # agreeing duplicate
            rows += [record("p1", "c1", drugs[1], "R")]  # conflicting key
            rows += [record("p1", "c1", drugs[2], "I")]  # S plus invalid key
            rows += [record("p1", "c1", drugs[3], "R", organism="K. pneumoniae")]
            rows += [record("p1", "c2", drugs[0], "R")]
            rows += [record(f"other{i}", f"event{i}", drugs[i % len(drugs)], "R" if i % 2 else "S")
                     for i in range(40)]
            rows += [record("bad_event", "bad", drugs[0]),
                     record("bad_event", "bad", drugs[1], culture_description="blood")]
            rows += [record("missing_org", "missing", drugs[0], organism=None)]
            write("cohort", rows)
            write("demographics", [
                {"anon_id": "p1", "order_proc_id_coded": "c1", "age": "20-29", "gender": "A"},
                {"anon_id": "p1", "order_proc_id_coded": "c1", "age": "30-39", "gender": "A"},
            ])
            write("ward_info", [
                {"anon_id": "p1", "order_proc_id_coded": "c1", "hosp_ward_IP": 0,
                 "hosp_ward_OP": 1, "hosp_ward_ER": 0, "hosp_ward_ICU": 0},
                {"anon_id": "p1", "order_proc_id_coded": "c1", "hosp_ward_IP": 0,
                 "hosp_ward_OP": 1, "hosp_ward_ER": 0, "hosp_ward_ICU": 0},
            ])
            write("labs", [
                {"anon_id": "p1", "order_proc_id_coded": "c1", "Period_Day": 14,
                 "median_wbc": 8., "median_cr": 1.},
                {"anon_id": "p1", "order_proc_id_coded": "c1", "Period_Day": 14,
                 "median_wbc": 8., "median_cr": 2.},
                {"anon_id": "p1", "order_proc_id_coded": "c1", "Period_Day": -14,
                 "median_wbc": 999., "median_cr": 999.},
            ])
            history_base = {"anon_id": "p1", "order_proc_id_coded": "c1",
                            "order_time_jittered_utc": "2020-01-01 00:00:00+00:00"}
            write("antibiotic_class_exposure", [
                dict(history_base, antibiotic_class="Penicillin", time_to_culturetime=10),
                dict(history_base, antibiotic_class="Penicillin", time_to_culturetime=10),
                dict(history_base, antibiotic_class="Cephalosporin", time_to_culturetime=0),
                dict(history_base, antibiotic_class="Carbapenem", time_to_culturetime=-10),
            ])
            write("microbial_resistance", [
                dict(history_base, antibiotic=drugs[0], resistant_time_to_culturetime=14),
                dict(history_base, antibiotic=drugs[0], resistant_time_to_culturetime=0),
                dict(history_base, antibiotic=drugs[4], resistant_time_to_culturetime=-1),
            ])
            frame, audit = build_dataset(raw, output, config)
            self.assertEqual(len(frame), 49)
            self.assertFalse(frame.duplicated(KEY).any())
            self.assertEqual(frame.groupby("patient_id")["split"].nunique().max(), 1)
            self.assertEqual(frame.groupby(LINK)["split"].nunique().max(), 1)
            self.assertEqual(set(frame["split"]), {"train", "validation", "test"})
            self.assertFalse(frame["patient_id"].isin(["bad_event", "missing_org"]).any())
            self.assertEqual(audit["key_label_decisions"]["conflicting_s_r"], 1)
            self.assertEqual(audit["key_label_decisions"]["invalid_or_unknown_result"], 1)
            self.assertEqual(audit["agreeing_duplicate_rows_collapsed"], 1)
            event = frame[(frame["patient_id"] == "p1") & (frame["culture_id"] == "c1")]
            self.assertEqual(len(event), 8)
            self.assertTrue(event["age_group"].isna().all())
            self.assertTrue(event["ward_ip"].eq(0).all())
            self.assertTrue(event["lab_median_wbc_14d"].eq(8).all())
            self.assertTrue(event["lab_median_creatinine_14d"].isna().all())
            self.assertTrue(event["prior_abx_days_ago"].eq(10).all())
            self.assertTrue(event["prior_abx_class_count_90d"].eq(1).all())
            candidate = event[event["candidate_antibiotic"] == drugs[0]]
            self.assertTrue(candidate["prior_candidate_resistance_days_ago"].eq(14).all())
            self.assertTrue(event.loc[event["candidate_antibiotic"] == drugs[4],
                                      "prior_candidate_resistance_days_ago"].isna().all())
            no_history = frame[frame["patient_id"].str.startswith("other")]
            self.assertTrue(no_history[NUMERIC_FEATURES].isna().all().all())
            self.assertTrue(no_history["ward_ip"].isna().all())
            numeric = _numeric_features(frame)
            self.assertEqual(numeric, NUMERIC_FEATURES)
            predictors = _feature_frame(frame, numeric)
            self.assertFalse({"label", "organism", "patient_id", "culture_id", "culture_time", "split"}
                             & set(predictors))
            profile_dataset(frame, output)
            summary = pd.read_csv(output / "cohort_summary.csv")
            self.assertEqual(summary.loc[summary["split"] == "all", "rows"].iloc[0], 49)
            self.assertTrue((output / "analytical.parquet").exists())
            pd.testing.assert_frame_equal(pd.read_parquet(output / "analytical.parquet"), frame)


if __name__ == "__main__":
    unittest.main()
