from io import open
import glob
import os
import unicodedata
import string

# print all file names
def findFiles(path) : return glob.glob(path)
# print(findFiles("data/names/*.txt"))

# printing all ascii letters
all_letters = string.ascii_letters + ".,;"
num_letters = len(all_letters)
# print(f"all letters : {all_letters}")
# print(f"total letters : {num_letters}")

# turning unicode strings to plain ascii
def unicodeToAscii(s) :
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
        and c in all_letters
    )
# print(unicodeToAscii("Ślusàrski"))

# building the category lines dictionary : 
# a list of names per language
category_lines = {}
all_categories = []

# reading a file and splitting in lines
def readLines(filename):
    lines = open(filename, encoding="utf-8").read().strip().split("\n")
    return [unicodeToAscii(line) for line in lines]


for filename in findFiles("data/names/*.txt"):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

# print(category_lines["Italian"][:5])