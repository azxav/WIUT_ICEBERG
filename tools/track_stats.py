"""Print track lifespan and sparse positions (1080p) for manual event review."""
import sys
import pandas as pd
for name, ids in [("C3896", [1754,3563,5818,5456,2604,7582]), ("C3897", [4,808,5731,1400,3459,3575,7616,81,1659,6166,8129]),
                  ("C3902", [8338,2533,2536,2493,5258,7815,7269]), ("C3905", [13,273,12])]:
    d = pd.read_parquet(f"cache/{name}_s3_960_yolo11s.parquet")
    for tid in ids:
        g=d[d.id==tid].sort_values('t')
        if not len(g): continue
        q=g.iloc[::max(1,len(g)//12)]
        print(name,tid,int(g.cls.mode().iloc[0]),round(g.t.min(),1),round(g.t.max(),1),
              [(round(r.t,1),round((r.x1+r.x2)/4),round(r.y2/2)) for r in q.itertuples()])
