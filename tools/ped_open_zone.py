"""Find pedestrian track presence in the open road between marked crossings."""
import pandas as pd
import numpy as np

for name in ["C3896","C3897","C3902","C3905"]:
 d=pd.read_parquet(f"cache/{name}_s3_960_yolo11s.parquet")
 d=d[d.cls==0].copy()
 d['x']=(d.x1+d.x2)/4;d['y']=d.y2/2
 # Central conflict area, away from the marked A/B/C zebra stripes.
 q=d[(d.x>770)&(d.x<1550)&(d.y>635)&(d.y<990)&(d.y>1150-.42*d.x)]
 rows=[]
 for tid,g in q.groupby('id'):
  if g.t.max()-g.t.min()>=1.2 and len(g)>=10:
   rows.append((g.t.min(),g.t.max(),tid,len(g),g.x.min(),g.x.max(),g.y.min(),g.y.max()))
 print('\n'+name)
 for a,b,tid,n,x0,x1,y0,y1 in sorted(rows):
  print(f'{a:6.1f}-{b:6.1f} id{tid:<5} n{n:<3} x{int(x0)}-{int(x1)} y{int(y0)}-{int(y1)}')
