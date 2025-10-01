import os
import pandas as pd

df = pd.read_csv("E:/Code/Contracting-FL-Benchmarking/resources/pacs_v1.0/metadata.csv")

missing = []
for p in df["path"].values:
    fpath = os.path.join("E:/Code/Contracting-FL-Benchmarking/resources/pacs_v1.0", p)
    if not os.path.exists(fpath):
        missing.append(fpath)

print(f"total {len(df)}")
print(f"total missing: {len(missing)}")
print("Examples of missing files:", missing[:10])
