"""Small adversarial source tables exercise history leakage and join safety."""
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from armd.support_features import NUMERIC_FEATURES, enrich_features


class SupportFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.raw = Path(self.temp.name)
        self.frame = pd.DataFrame({
            "patient_id": ["p1", "p1", "p2"], "culture_id": ["c1", "c1", "c2"],
            "culture_time": pd.to_datetime(["2020-01-01"] * 3, utc=True),
            "candidate_antibiotic": ["ceftriaxone", "ciprofloxacin", "ceftriaxone"],
            "organism": ["current organism"] * 3, "label": [0, 1, 0],
        }, index=[10, 30, 50])

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, records):
        pd.DataFrame(records).to_csv(self.raw / ("microbiology_cultures_" + name + ".csv"), index=False)

    def test_missing_sources_leave_all_features_missing(self):
        audit = {}
        result = enrich_features(self.frame, self.raw, audit)
        pd.testing.assert_frame_equal(result[self.frame.columns], self.frame)
        self.assertTrue(result[NUMERIC_FEATURES].isna().all().all())
        self.assertEqual(audit["support_features"]["input_rows"], 3)

    def test_missing_optional_sources_preserve_nullable_string_keys(self):
        typed = self.frame.astype({"patient_id": "string", "culture_id": "string",
                                   "candidate_antibiotic": "string"})
        for source in [typed, typed.iloc[:0]]:
            with self.subTest(rows=len(source)):
                result = enrich_features(source, self.raw, {})
                pd.testing.assert_frame_equal(result[source.columns], source)
                self.assertTrue(result[NUMERIC_FEATURES].isna().all().all())

    def test_duplicate_summaries_do_not_average_or_expand_grain(self):
        self.write("labs", [
            {"anon_id": "p1", "order_proc_id_coded": "c1", "Period_Day": 14,
             "median_wbc": 10, "median_cr": 1.1},
            {"anon_id": "p1", "order_proc_id_coded": "c1", "Period_Day": 14,
             "median_wbc": 10, "median_cr": 1.4},
            {"anon_id": "p1", "order_proc_id_coded": "c1", "Period_Day": -14,
             "median_wbc": 999, "median_cr": 999},
            {"anon_id": "p2", "order_proc_id_coded": "c2", "Period_Day": 14,
             "median_wbc": "Null", "median_cr": "Null"},
        ])
        self.write("vitals", [
            {"anon_id": "p1", "order_proc_id_coded": "c1", "median_heartrate": 70,
             "median_temp": "Null"},
            {"anon_id": "p1", "order_proc_id_coded": "c1", "median_heartrate": 70,
             "median_temp": 37},
        ])
        audit = {}
        result = enrich_features(self.frame, self.raw, audit)
        self.assertEqual(len(result), 3)
        self.assertEqual(result.loc[10, "lab_median_wbc_14d"], 10)
        self.assertTrue(pd.isna(result.loc[10, "lab_median_creatinine_14d"]))
        self.assertEqual(result.loc[30, "vital_median_temperature_48h"], 37)
        self.assertTrue(result.loc[50, NUMERIC_FEATURES].isna().all())
        labs = audit["support_features"]["tables"]["microbiology_cultures_labs.csv"]
        self.assertEqual(labs["conflicting_cells_set_missing"], 1)
        self.assertEqual(labs["undocumented_window_rows_excluded"], 1)

    def test_history_strict_cutoff_timestamp_match_and_candidate_matching(self):
        base = {"anon_id": "p1", "order_proc_id_coded": "c1",
                "order_time_jittered_utc": "2020-01-01 00:00:00+00:00"}
        self.write("antibiotic_class_exposure", [
            dict(base, antibiotic_class="Penicillin", time_to_culturetime=90),
            dict(base, antibiotic_class="penicillin", time_to_culturetime=90),
            dict(base, antibiotic_class="Cephalosporin", time_to_culturetime=10),
            dict(base, antibiotic_class="Carbapenem", time_to_culturetime=91),
            dict(base, antibiotic_class="Future", time_to_culturetime=-1),
            dict(base, antibiotic_class="Same-day", time_to_culturetime=0),
            dict(base, antibiotic_class="Missing lag", time_to_culturetime="Null"),
            dict(base, antibiotic_class="Wrong event time", time_to_culturetime=1,
                 order_time_jittered_utc="2021-01-01 00:00:00+00:00"),
        ])
        self.write("microbial_resistance", [
            dict(base, antibiotic="Ceftriaxone", resistant_time_to_culturetime=60),
            dict(base, antibiotic="CEFTRIAXONE", resistant_time_to_culturetime=30),
            dict(base, antibiotic="Ceftriaxone", resistant_time_to_culturetime=0),
            dict(base, antibiotic="Ciprofloxacin", resistant_time_to_culturetime=-5),
        ])
        audit = {}
        result = enrich_features(self.frame, self.raw, audit)
        self.assertEqual(result.loc[10, "prior_abx_days_ago"], 10)
        self.assertEqual(result.loc[30, "prior_abx_class_count_90d"], 2)
        self.assertEqual(result.loc[10, "prior_candidate_resistance_days_ago"], 30)
        self.assertTrue(pd.isna(result.loc[30, "prior_candidate_resistance_days_ago"]))
        self.assertTrue(result.loc[50, NUMERIC_FEATURES].isna().all())
        # Current outcomes and organism identity do not enter these features.
        changed = self.frame.assign(label=1-self.frame.label, organism="unseen current organism")
        changed_result = enrich_features(changed, self.raw, {})
        pd.testing.assert_frame_equal(result[NUMERIC_FEATURES], changed_result[NUMERIC_FEATURES])
        table = audit["support_features"]["tables"]["microbiology_cultures_antibiotic_class_exposure.csv"]
        self.assertEqual(table["same_day_or_future_rows_excluded"], 2)
        self.assertEqual(table["event_timestamp_mismatch_or_missing_rows_excluded"], 1)

    def test_metadata_covers_every_feature(self):
        path = Path(__file__).parents[1] / "src/armd/support_feature_spec.json"
        spec = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(spec["numeric_features"], NUMERIC_FEATURES)
        self.assertEqual([item["name"] for item in spec["features"]], NUMERIC_FEATURES)


if __name__ == "__main__":
    unittest.main()
