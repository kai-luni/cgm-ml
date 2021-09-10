from pathlib import Path

from cgmml.common.evaluation.CV.csv_utils import read_csv
from cgmml.common.evaluation.CV.evaluation import check_height_prediction, filter_metadata, update_output

TESTS_PATH = Path(__file__).parents[0].absolute()
METADATA_FILE = TESTS_PATH / 'test_metadata.csv'
METADATA_ERROR = 7


def test_check_height_prediction():

    # Test low value exception
    try:
        check_height_prediction(0)
        assert False
    except Exception:
        assert True

    # Test high value exception
    try:
        check_height_prediction(200)
        assert False
    except Exception:
        assert True

    # Test valid value pass
    try:
        check_height_prediction(100)
        assert True
    except Exception:
        assert False


def test_filter_metadata():

    # Check standing children scans
    assert len(filter_metadata(read_csv(METADATA_FILE), True, False)) == 3

    # Check lying children scans
    assert len(filter_metadata(read_csv(METADATA_FILE), False, False)) == 0


def test_update_output():
    sum_err = 0
    output = []
    rejections = []
    angles = [0, 0, 0]
    heights = [60, 80, 200]
    data = [0, 0, 0, 0, 0, 79]
    update_output(angles, heights, '', data, output, rejections, True, sum_err)

    # Check processed scans
    assert len(output) == 1

    # Check rejected scans
    assert len(rejections) == 0

    # Check error abs(80 - 79) == 1
    assert data[METADATA_ERROR] == 1


if __name__ == "__main__":
    test_check_height_prediction()
    test_filter_metadata()
    test_update_output()
