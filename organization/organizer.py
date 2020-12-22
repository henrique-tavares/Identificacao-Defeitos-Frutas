from os import walk, mkdir, path, sep as os_sep
from shutil import move
import csv

fruit_type = "cajus-amarelos"

with open(path.join(path.curdir, "classification", f"{fruit_type}.csv")) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    csv_rows = [row for row in csv_reader]

label_transform = {"A": "good", "B": "medium", "C": "bad"}

for quality_class in label_transform.values():
    try:
        mkdir(path.join(path.curdir, "..", "databases", fruit_type, quality_class))
    except:
        pass

for filename, label in csv_rows:
    label = label_transform.get(label)

    for root, dirs, files in walk(path.join(path.curdir, "..", "databases", fruit_type)):
        if root.split(os_sep)[-1] == label:
            continue

        elif filename in files:
            move(path.join(root, filename), path.join(path.curdir, "..", "databases", fruit_type, label))
            print(
                "moving:",
                path.join(root, filename),
                "to:",
                path.join(path.curdir, "..", "databases", fruit_type, label),
            )
