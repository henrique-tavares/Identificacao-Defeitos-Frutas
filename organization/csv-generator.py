from os import path, listdir

fruit_type = "cajus-vermelhos"

filenames = listdir(path.join(path.curdir, "..", "databases", fruit_type))
filenames = list(filter(lambda filename: filename.endswith(".png"), filenames))

try:
    with open(path.join(path.curdir, "classification", f"{fruit_type}.csv"), "x") as csv:
        csv.write("image,label\n" + "\n".join((filename + "," for filename in sorted(filenames))))

except FileExistsError:
    with open(path.join(path.curdir, "classification", f"{fruit_type}.csv"), "a") as csv:
        csv.write("\n" + "\n".join((filename + "," for filename in sorted(filenames))))
