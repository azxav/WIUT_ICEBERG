"""Near-approach stopped/total vehicle counts per second for congestion review."""
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.scene import load_scene

for name in ["C3896","C3897","C3902","C3905"]:
 d=pd.read_parquet(f'cache/{name}_s3_960_yolo11s.parquet')
 d=d[d.cls.isin([1,2,3,5,7])].sort_values(['id','t']).copy()
 d['x']=(d.x1+d.x2)/4; d['y']=d.y2/2
 for col in ['x','y','t']:
  d[f'old_{col}']=d.groupby('id')[col].shift(10)
 d['speed']=np.hypot(d.x-d.old_x,d.y-d.old_y)/(d.t-d.old_t)
 scene,_=load_scene(cv2.imread(f'eda/frames/{name}_background.jpg'))
 poly=np.float32(scene['zones']['queue_near']['polygon'])
 d=d[[cv2.pointPolygonTest(poly,(float(x),float(y)),False)>=0 for x,y in zip(d.x,d.y)]]
 d['sec']=d.t.round().astype(int)
 g=d.groupby('sec').agg(total=('id','nunique'),stationary=('speed',lambda x:int((x<5).sum()/9)))
 print('\n'+name)
 for t in range(0,int(d.t.max())+1,5):
  if t in g.index: print(f'{t:3d}:{int(g.loc[t,"stationary"]):2d}/{int(g.loc[t,"total"]):2d}',end=' ')
 print()
