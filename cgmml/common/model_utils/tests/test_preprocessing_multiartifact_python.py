import pytest

from cgmml.common.model_utils.preprocessing_multiartifact_python import (
    sample_systematic_from_artifacts, sample_windows_from_artifacts, REGEX_PICKLE)


def test_sample_windows_from_artifacts_multiple_results():
    artifacts = ['001.p', '002.p', '003.p', '004.p', '005.p', '006.p']
    actual = list(sample_windows_from_artifacts(artifacts, 5))
    expected = [
        ['001.p', '002.p', '003.p', '004.p', '005.p'],
        ['002.p', '003.p', '004.p', '005.p', '006.p'],
    ]
    assert actual == expected


def test_sample_windows_from_artifacts_one_result():
    artifacts = ['001.p', '002.p', '003.p', '004.p', '005.p']
    actual = list(sample_windows_from_artifacts(artifacts, 5))
    expected = [
        ['001.p', '002.p', '003.p', '004.p', '005.p'],
    ]
    assert actual == expected


def test_sample_windows_from_artifacts_no_result():
    artifacts = ['001.p', '002.p', '003.p', '004.p']
    actual = list(sample_windows_from_artifacts(artifacts, 5))
    assert actual == []


def test_systematic_sample_from_many_artifacts():
    artifacts = list(range(20, 0, -1))
    n_artifacts = 5
    selected_artifacts = sample_systematic_from_artifacts(artifacts, n_artifacts)
    assert selected_artifacts == [18, 14, 10, 6, 2]
    assert len(selected_artifacts) == n_artifacts


def test_systematic_sample_from_few_artifacts():
    artifacts = ['0', '1', '2', '3', '4', '5', '6']
    n_artifacts = 5
    selected_artifacts = sample_systematic_from_artifacts(artifacts, n_artifacts)
    assert selected_artifacts[0] == '0'
    assert selected_artifacts[4] == '4'
    assert len(selected_artifacts) == n_artifacts


def test_systematic_sample_from_artifacts_too_few():
    artifacts = list(range(3, 0, -1))
    n_artifacts = 5
    with pytest.raises(Exception):
        sample_systematic_from_artifacts(artifacts, n_artifacts)


def test_regex_pickle():
    fname = "pc_00c38d20-0d16-11ec-985b-cb6242a5c193_2021-09-04-00-21-04-443303_100_1.0.p"

    match_result = REGEX_PICKLE.search(fname)
    assert match_result.group("person_id") == "00c38d20-0d16-11ec-985b-cb6242a5c193"
    assert match_result.group("timestamp") == "2021-09-04-00-21-04-443303"
    assert match_result.group("code") == "100"
    assert match_result.group("idx") == "1.0"
