import pandas as pd
for name in ["C3896","C3897","C3902","C3905"]:
 c=pd.read_csv(f"cache/{name}_candidates.csv")
 d=pd.read_parquet(f"cache/{name}_s3_960_yolo11s.parquet")
 c=c[(c.kind=='jaywalking')&(c.end-c.start>=2)]
 rows=[]
 for r in c.itertuples():
  g=d[(d.id==r.track)&d.t.between(r.start-.1,r.end+.1)]
  if len(g)<3:continue
  x=(g.x1+g.x2)/4;y=g.y2/2
  disp=((x.iloc[-1]-x.iloc[0])**2+(y.iloc[-1]-y.iloc[0])**2)**.5
  rows.append((r.start,r.end,int(r.track),int(disp),int(x.iloc[0]),int(y.iloc[0]),int(x.iloc[-1]),int(y.iloc[-1])))
 print('\n'+name)
 for a,b,tid,disp,x0,y0,x1,y1 in sorted(rows,key=lambda x:-x[3])[:50]:
  print(f'{a:6.1f}-{b:6.1f} id{tid:<5} d{disp:<4} ({x0},{y0})->({x1},{y1})')
