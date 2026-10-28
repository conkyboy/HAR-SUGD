# tie_report.py -- run after run_audit_ties.py
import json
rows = json.load(open("tie_audit.json"))
tot  = sum(len(r["rounds"]) for r in rows)
tie  = sum(x["n_tied_exact"] > 1 for r in rows for x in r["rounds"])
chg  = sum(x["tie_changed_choice"] for r in rows for x in r["rounds"])
diff = sum(not r["same_reduct"] for r in rows)
print(f"rounds={tot}  tie-rounds={tie} ({tie/tot:.1%})  "
      f"GD-changed={chg}  differing-reducts={diff}/{len(rows)}")
by_ds = {}
for r in rows:
    d = by_ds.setdefault(r["dataset"],
                         {"rounds": 0, "ties": 0, "chg": 0, "diff": 0, "len": []})
    d["rounds"] += len(r["rounds"])
    d["ties"]  += sum(x["n_tied_exact"] > 1 for x in r["rounds"])
    d["chg"]   += sum(x["tie_changed_choice"] for x in r["rounds"])
    d["diff"]  += 0 if r["same_reduct"] else 1
    d["len"].append(len(r["red_gd"]))
print(f"{'ds':>6s}  {'|R|':>5s}  ties/rounds  changed  reduct_diff")
for k in sorted(by_ds):
    v = by_ds[k]
    print(f"{k:>6s}  {sum(v['len'])/len(v['len']):5.1f}  "
          f"{v['ties']:4d}/{v['rounds']:4d}  {v['chg']:5d}  {v['diff']}/10")