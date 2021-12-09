import sys

from csv_utils import read_csv, write_csv

METADATA_PERSON_ID = 0
METADATA_SCAN_ID = 1
METADATA_ORDER = 2
METADATA_ARTIFACT_ID = 3
METADATA_FILEPATH = 4
METADATA_EXTENSION = 5
METADATA_MANUAL_HEIGHT = 6
METADATA_MANUAL_DATE = 11

MAXIMAL_HEIGHT_DAILY_DIFF = 2.0
MAXIMAL_HEIGHT_MEASURES_DIFF = 5.0


def blacklist_invalid(indata):

    # Get unique persons
    persons = {}
    size = len(indata)
    for index in range(1, size):
        data = indata[index]
        person = data[METADATA_PERSON_ID].replace('"', '')
        if not persons.get(person):
            persons[person] = [data]
        else:
            persons[person].append(data)

    blacklist = {}
    for person in persons:

        # Blacklist data where there are on one day different height measures
        growth = {}
        for data in persons[person]:
            date = data[METADATA_MANUAL_DATE][0:10]
            height = float(data[METADATA_MANUAL_HEIGHT])
            if not growth.get(date):
                growth[date] = height
            elif abs(growth[date] - height) > MAXIMAL_HEIGHT_DAILY_DIFF:
                blacklist[person] = True
                break

        # Blacklist data where there are too big differences between measures
        last = 0
        for date in growth:
            height = growth[date]
            if last == 0:
                last = height
            elif abs(height - last) > MAXIMAL_HEIGHT_MEASURES_DIFF:
                blacklist[person] = True
                break

    # Recreate the structure without blacklisted persons
    outdata = [indata[0]]
    for index in range(1, size):
        data = indata[index]
        person = data[METADATA_PERSON_ID].replace('"', '')
        if not blacklist.get(person):
            outdata.append(data)
    return outdata


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print('You did not enter metadata file path')
        print('E.g.: python rgbd_match.py metadata_path')
        sys.exit(1)

    metadata_file = sys.argv[1]

    # Create a map
    indata = read_csv(metadata_file)
    indata = blacklist_invalid(indata)
    size = len(indata)
    mapping = {}
    for index in range(1, size):
        data = indata[index]
        if data[METADATA_EXTENSION] != '.jpg':
            continue
        scanid = data[METADATA_SCAN_ID]
        order = data[METADATA_ORDER]
        key = scanid + str(order)
        mapping[key] = data

    # For every depthmap add rgb file
    output = []
    processed = {}
    for index in range(1, size):
        data = indata[index]
        if data[METADATA_EXTENSION] != '.depth':
            continue
        scanid = data[METADATA_SCAN_ID]
        order = data[METADATA_ORDER]
        key = scanid + str(order)
        if processed.get(key):
            continue

        # add the RGB-D match
        if mapping.get(key):
            data[METADATA_EXTENSION] = mapping[key][METADATA_FILEPATH]
            del data[0]
            output.append(data)
            processed[key] = True
    write_csv('newmetadata.csv', output)
