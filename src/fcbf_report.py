# fcbf_report.py -- summarize FCBF run; paste the output back
import json
import pandas as pd

df = pd.read_csv("fcbf_raw.csv")
summ = df.groupby(["clf", "abbr"]).agg(
    acc=("acc", "mean"), std=("acc", "std"), n_sel=("n_sel", "mean")).reset_index()
for clf in ["KNN", "CART"]:
    t = summ[summ.clf == clf].set_index("abbr")[["acc", "std", "n_sel"]]
    print(f"\n===== FCBF {clf}: acc / std / |S| =====")
    print(t.round(3).to_string())

print("\nFCBF total selection time over 18x10 folds (s):",
      round(df.groupby(["abbr", "fold"]).t.first().sum(), 1))

sw = pd.DataFrame(json.load(open("fcbf_delta.json")))
print("\n===== mean |S| by delta =====")
print(sw.groupby(["abbr", "delta"]).n_sel.mean().unstack().round(1).to_string())

main = df.drop_duplicates(["abbr", "fold"]).set_index(["abbr", "fold"])["sel"]
sw["same_as_0.5"] = sw.apply(
    lambda r: str(r["sel"]) == main.get((r["abbr"], r["fold"])), axis=1)
print("\nselection identical to delta=0.5:")
print(sw.groupby("delta")["same_as_0.5"].agg(["sum", "count"]).to_string())

g = df.drop_duplicates(["abbr", "fold"])["guard"]
print(f"\nguard triggered at delta=0.5 (empty pre-filter): {int(g.sum())}/180")
print("datasets affected:", sorted(df[df.guard].abbr.unique().tolist()))