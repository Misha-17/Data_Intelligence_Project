import pandas as pd
from armd.build import canonical_drug, collapse_support, patient_split, resolve_labels, sample_patients


def labels(values):
    return pd.DataFrame({'patient_id': ['p1']*len(values), 'culture_id': ['c1']*len(values),
                         'organism': ['e coli']*len(values), 'candidate_antibiotic': ['ampicillin']*len(values),
                         'susceptibility': values})


def test_conflict_drops_whole_key_not_majority_vote():
    retained, audit = resolve_labels(labels(['S', 'S', 'R']))
    assert retained.empty
    assert audit.reason.iloc[0] == 'conflicting_s_r'


def test_intermediate_or_unknown_with_valid_result_still_excludes_key():
    for value in ['Intermediate', 'Susceptible - Dose Dependent', None, 'Inconclusive', 'Synergism']:
        retained, _ = resolve_labels(labels(['Susceptible', value]))
        assert retained.empty


def test_agreeing_labels_collapse_and_aliases_standardize():
    retained, _ = resolve_labels(labels(['Susceptible', 's']))
    assert retained.label.tolist() == [1]
    assert canonical_drug('Amoxicillin/Clavulanic Acid') == 'amoxicillin/clavulanate'


def test_conflicting_support_is_missing_and_never_multiplies_join():
    support = pd.DataFrame({'patient_id': ['p1','p1','p2'], 'culture_id': ['c1','c1','c2'],
                            'age_group': ['18-24','25-34',None], 'gender_code': ['0','0',None]})
    audit = {}
    result = collapse_support(support, ['age_group','gender_code'], audit, 'demo')
    assert len(result) == 2
    assert result.age_group.isna().all()
    assert result.gender_code.iloc[0] == '0'
    assert audit['demo']['conflicting_cells_set_missing'] == 1


def test_patient_split_and_sample_do_not_depend_on_order_or_duplicate_rows():
    patients = pd.Series([f'p{i}' for i in range(100)] + ['p4']*5)
    split = patient_split(patients)
    assert split[patients.eq('p4')].nunique() == 1
    assert split.to_dict() == patient_split(patients.sample(frac=1, random_state=2)).sort_index().to_dict()
    chosen = sample_patients(patients, .25, 42)
    assert chosen[patients.eq('p4')].nunique() == 1
