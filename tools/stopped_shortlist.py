import pandas as pd
for name in ['C3896','C3897','C3902','C3905']:
 c=pd.read_csv(f'cache/{name}_candidates.csv');d=pd.read_parquet(f'cache/{name}_s3_960_yolo11s.parquet')
 q=c[(c.kind=='stopped_or_queue')&(c.end-c.start>=9)]
 rows=[]
 for r in q.itertuples():
  g=d[(d.id==r.track)&d.t.between(r.start+.5,r.end-.5)]
  if len(g)==0:continue
  x=int((g.x1+g.x2).median()/4);y=int(g.y2.median()/2)
  rows.append((r.start,r.end,int(r.track),x,y))
 print('\n'+name)
 print([(round(a,1),round(b,1),tid,x,y) for a,b,tid,x,y in rows if (x>920 or y>620)])
