"""Municipal population 2005-2042 (total and urban) from the DANE projection workbook.

Run from the project root:
    python scripts/01_pop.py
"""
import collections
import csv
import pathlib

import openpyxl

ROOT = pathlib.Path(__file__).resolve().parents[1]
(ROOT / "results").mkdir(exist_ok=True)
p = ROOT / "inputs" / "population_2005-2042.xlsx"
wb=openpyxl.load_workbook(p, read_only=True, data_only=True)
ws=wb['Sheet1']
it=ws.iter_rows(values_only=True)
hdr=list(next(it))
idx_m=hdr.index('municipality_code'); idx_y=hdr.index('year'); idx_a=hdr.index('geographic_area')
val=[i for i,h in enumerate(hdr) if isinstance(h,str) and (h.startswith('men_') or h.startswith('women_'))]
tot=collections.defaultdict(float); urb=collections.defaultdict(float)
n=0
for row in it:
    n+=1
    m=str(row[idx_m]).zfill(5); y=int(row[idx_y]); a=str(row[idx_a]).lower()
    s=0.0
    for i in val:
        v=row[i]
        if v is not None:
            try: s+=float(v)
            except: pass
    tot[(m,y)]+=s
    if 'urban' in a or 'cabec' in a: urb[(m,y)]+=s
print("rows:",n)
with open(ROOT / 'results' / 'population_municipal.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['municipality_code','year','pop_total','pop_urban'])
    for (m,y),v in sorted(tot.items()): w.writerow([m,y,round(v,1),round(urb.get((m,y),0.0),1)])
print("written")
