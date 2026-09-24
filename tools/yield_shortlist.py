import pandas as pd
for name in ["C3896","C3897","C3902","C3905"]:
 c=pd.read_csv(f'cache/{name}_candidates.csv')
 p=c[c.kind=='crosswalk_person'];v=c[c.kind=='crosswalk_vehicle']
 hits=[]
 for z in v.itertuples():
  pp=p[(p.detail==z.detail)&(p.start<=z.end)&(p.end>=z.start)]
  if len(pp):
   hits.append((round(z.start,1),round(z.end,1),z.detail,int(z.track),','.join(map(str,pp.track.unique()))))
 print('\n',name,len(hits))
 print(hits[:120])
