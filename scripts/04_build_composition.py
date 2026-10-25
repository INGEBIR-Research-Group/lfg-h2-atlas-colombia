"""Waste composition for the landfill-gas hydrogen atlas.

Two tiers: the municipality's own measured characterization (PGIRS) and, where its plan reports none, the UPME departmental profile. UPME (2018) Table 7 lists all 32 departments and Bogota D.C., so every municipality resolves to a departmental profile and no national average is used; the script stops if a department lacks a profile. Writes

    results/composition_municipality.csv   one row per served municipality
    results/composition_site.csv           canonical site x year, 2010-2024
    results/composition_stats.json         coverage statistics

from the measured characterizations and the UPME table in inputs/composition/.

Run from the project root after scripts/03_build_waste.py:
    python scripts/04_build_composition.py
"""
import json
import pathlib
import re
import unicodedata

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
HERE = ROOT / "inputs" / "composition"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def norm(s):
    if pd.isna(s): return ""
    s=str(s).upper().strip()
    s=''.join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c)!='Mn')
    s=re.sub(r'[^A-Z ]',' ',s); s=re.sub(r'\s+',' ',s).strip()
    return s

DEPT_FIX={'BOGOTA D C':'BOGOTA','ARCHIPIELAGO DE SAN ANDRES PROVIDENCIA Y SANTA CATALINA':'SAN ANDRES Y PROVIDENCIA',
          'SAN ANDRES':'SAN ANDRES Y PROVIDENCIA','GUAJIRA':'LA GUAJIRA','NARINO':'NARINO','VALLE':'VALLE DEL CAUCA'}
def dnorm(s):
    n=norm(s); return DEPT_FIX.get(n,n)

FRACS=['food','garden','paper','wood','textiles','diapers','nondeg']
L0C={'c1':70.0,'c2':113.0,'c3':186.0,'c4':200.0}
FOOD_SHARE=0.860           # UPME: food is 52.52 of the 61.06 percentage points of category 1
# Category 10 of UPME mixes nappies and sanitary paper with other inorganic residue. Nappies are taken as 5.5/7.8 of it, the ratio of the national nappy share of the 2016 policy characterization to the national share of category 10.
NAPPY_SHARE_C10=0.055/0.078
INORG=['c6_metals','c7_cd','c8_glass','c9_plastic','c10_diapers_other']

# ---------- Tier 2: UPME departmental profiles ----------
u=pd.read_csv(HERE/'upme_dept_composition.csv')
u=u[u['department'].str.lower()!='departamento'].copy()
u['dept_n']=u['department'].map(dnorm)
num=u.columns[1:11]
u[num]=u[num].astype(float)
# a department that reports neither plastic nor glass has not itemized its inorganic residue, which then sits in category 10; its nappy share is scaled from the itemized departments by the size of the inorganic residue
u['complete']=(u['c8_glass']>0)&(u['c9_plastic']>0)
u['inorg']=u[INORG].sum(axis=1)
distinct=u[u['complete']].drop_duplicates(subset=list(num))
NAPPY_MEAN=float((NAPPY_SHARE_C10*distinct['c10_diapers_other']).mean())
INORG_MEAN=float(distinct['inorg'].mean())
prof={}
for _,r in u.iterrows():
    c1,c2,c3,c4,c5,c6,c7,c8,c9,c10=[float(r[c]) for c in num]
    nap=NAPPY_SHARE_C10*c10 if r['complete'] else min(c10, NAPPY_MEAN*r['inorg']/INORG_MEAN)
    v={'food':FOOD_SHARE*c1,'garden':(1-FOOD_SHARE)*c1,'paper':c2,'wood':c3+c4,
       'textiles':c5,'diapers':nap,'nondeg':c6+c7+c8+c9+(c10-nap)}
    s_=sum(v.values()); prof[r['dept_n']]={k:100*x/s_ for k,x in v.items()}
print(f"nappy share: {100*NAPPY_SHARE_C10:.1f} % of category 10 in itemized departments; "
      f"mean {NAPPY_MEAN:.2f} % of waste against an inorganic residue of {INORG_MEAN:.1f} %")
print("departments scaled:", ", ".join(u.loc[~u['complete'],'department']))

# ---------- Tier 1: measured PGIRS compositions ----------
# one row per municipality with a measured characterization, percent by wet mass; a blank cell is a fraction that the plan does not report. Rubber, leather and bone, which a plan reports within its residual, belong to the slow decay category, as in the departmental tier
m=pd.read_csv(HERE/"msw_composition_measured.csv",dtype=str,keep_default_na=False)
measured={}
SLOW_MOVED={}
for _,r in m.iterrows():
    dn=dnorm(r['department']); mn=norm(r['municipality'])
    vals={}; miss=[]
    for k in FRACS:
        if r[k]=='': miss.append(k)
        else: vals[k]=float(r[k])
    lump=vals.pop('nondeg',0.0); miss=[k for k in miss if k!='nondeg']
    slow=float(r['rubber_leather_bone']) if r['rubber_leather_bone'] else 0.0
    if slow>0:
        vals['wood']=vals.get('wood',0.0)+slow
        SLOW_MOVED[mn]=round(slow,2)
    if dn not in prof:
        raise SystemExit(f"no departmental profile for {r['department']} ({r['municipality']})")
    p=prof[dn]
    # split the residual lump between the unreported degradable fractions and the true inerts, in the proportions of the departmental profile
    keys=miss+['nondeg']; w=np.array([p[k] for k in keys],dtype=float)
    w=w/w.sum() if w.sum()>0 else np.ones(len(keys))/len(keys)
    for k,wi in zip(keys,w): vals[k]=vals.get(k,0.0)+lump*wi
    s=sum(vals.values())
    measured[(dn,mn)]={k:100*vals.get(k,0.0)/s for k in FRACS}
print("measured municipalities usable:",len(measured))
print("rubber, leather and bone moved to the slow category:",SLOW_MOVED)

def L0(v):
    c1=v['food']+0.20*v['diapers']; c2=v['garden']
    c3=v['paper']+v['textiles']+0.80*v['diapers']; c4=v['wood']
    return (c1*L0C['c1']+c2*L0C['c2']+c3*L0C['c3']+c4*L0C['c4'])/100.0

# ---------- Municipality-level matrix ----------
lf=pd.read_csv(ROOT/"inputs"/"landfilling_site_code_imputed.csv",dtype=str)
lf['avg_tons_per_year']=pd.to_numeric(lf['avg_tons_per_year'],errors='coerce')
lf['year']=pd.to_numeric(lf['year'],errors='coerce')
lf['dept_n']=lf['served_department'].map(dnorm); lf['mun_n']=lf['served_municipality'].map(norm)

muns=lf[['served_dane_code','served_department','served_municipality','dept_n','mun_n']].drop_duplicates('served_dane_code')
rows=[]
for _,r in muns.iterrows():
    key=(r['dept_n'],r['mun_n'])
    if key in measured: v,tier=measured[key],'measured'
    elif r['dept_n'] in prof: v,tier=prof[r['dept_n']],'departmental'
    else: raise SystemExit(f"no departmental profile for {r['served_department']} ({r['served_municipality']})")
    rows.append(dict(municipality_code=r['served_dane_code'],department=r['served_department'],
                     municipality=r['served_municipality'],tier=tier,
                     **{k:round(v[k],3) for k in FRACS}, L0_m3CH4_per_Mg=round(L0(v),2)))
mu=pd.DataFrame(rows).sort_values('municipality_code')
print("\nmunicipalities:",len(mu)); print(mu['tier'].value_counts().to_string())
print("L0 by tier:\n",mu.groupby('tier')['L0_m3CH4_per_Mg'].describe()[['count','mean','min','max']].round(1).to_string())

# ---------- Site-level tonnage-weighted composition ----------
mu.to_csv(OUT/'composition_municipality.csv',index=False,encoding='utf-8-sig')
w=pd.read_csv(ROOT/'results'/'waste_municipality_site_year.csv',dtype={'served_dane_code':str})
w=w[w['year'].between(2010,2024)]
w=w.merge(mu[['municipality_code','tier']+FRACS],left_on='served_dane_code',right_on='municipality_code',how='left')
if w[FRACS].isna().any().any():
    raise SystemExit('served municipalities without a composition')
w['measured_t']=np.where(w['tier']=='measured',w['tons'],0.0)
agg={k:(k,'sum') for k in []}
for k in FRACS: w[k+'_t']=w[k]*w['tons']
g=w.groupby(['site_id','year']).agg(tons=('tons','sum'),measured_t=('measured_t','sum'),
                                    n_municipalities=('served_dane_code','nunique'),
                                    **{k+'_t':(k+'_t','sum') for k in FRACS}).reset_index()
for k in FRACS: g[k]=(g[k+'_t']/g['tons']).round(3)
g['share_measured']=(100*g['measured_t']/g['tons']).round(1)
g['L0_m3CH4_per_Mg']=[round(L0({k:r[k] for k in FRACS}),2) for _,r in g.iterrows()]
site=g[['site_id','year','n_municipalities','tons','share_measured']+FRACS+['L0_m3CH4_per_Mg']]
site=site.sort_values(['year','tons'],ascending=[True,False])
site.to_csv(OUT/'composition_site.csv',index=False,encoding='utf-8-sig',float_format='%.3f')
print("\nsite-year records:",len(site)," unique sites:",site['site_id'].nunique())
tw=np.average(site['L0_m3CH4_per_Mg'],weights=site['tons'])
print(f"tonnage-weighted national L0 = {tw:.1f} m3 CH4/Mg")
sm_=np.average(site['share_measured'],weights=site['tons'])
print(f"tonnage share of disposal covered by a measured composition = {sm_:.1f} %")
lo,hi=site['L0_m3CH4_per_Mg'].min(),site['L0_m3CH4_per_Mg'].max()
json.dump({'L0_weighted':round(float(tw),2),'share_measured':round(float(sm_),1),
           'L0_min':float(lo),'L0_max':float(hi),'nappy_share_c10':NAPPY_SHARE_C10,
           'nappy_mean_itemized':NAPPY_MEAN,'inorganic_mean_itemized':INORG_MEAN,
           'n_measured':int((mu.tier=='measured').sum()),'n_sites':int(site['site_id'].nunique()),
           'slow_moved':dict(sorted(SLOW_MOVED.items())),'n_slow_moved':len(SLOW_MOVED)},
          open(OUT/'composition_stats.json','w'),indent=1)
print("\nTop 12 sites in 2024 by tonnage:")
print(site[site.year==2024].head(12)[['site_id','tons','n_municipalities','share_measured','L0_m3CH4_per_Mg']].to_string(index=False))
