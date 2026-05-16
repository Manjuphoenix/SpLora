import pandas as pd

df = pd.read_csv("indraeye.csv")

result = df.groupby("name").agg({
    "similarity": ["max", "mean"],   # compute both
    "occurrence": "sum"
}).reset_index()

# flatten cols
result.columns = ["name", "max_similarity", "mean_similarity", "occurrence"]

# sort by mean similarity high → low
result = result.sort_values(by="mean_similarity", ascending=False)

print(result)

