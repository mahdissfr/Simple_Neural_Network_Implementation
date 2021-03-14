import csv


def read_from_file():
    data = []
    with open('data.csv') as csvfile:
        read = csv.reader(csvfile, delimiter=',')
        for row in read:
            data.append([float(row[0]),float(row[1]), float(row[2])])
    return data
