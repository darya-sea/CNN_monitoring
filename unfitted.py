import pandas as pd
from pprint import pprint

data = dict(pd.read_csv("Unfitted.csv", dtype=object).groupby('Folder')['Name'].apply(list))

cn =  0

for d in data:
    cn += len(d)

print(cn)