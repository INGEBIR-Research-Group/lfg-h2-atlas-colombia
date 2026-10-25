"""Cost-capacity dataset for reformer islands producing hydrogen from methane. Every row is transcribed from a published source. Normalisation to a common basis happens in capex_regression.py, never here."""
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
(ROOT / "results").mkdir(exist_ok=True)

# CEPCI annual averages, Chemical Engineering. Costs are expressed in USD of 2023, the reference year of the study; costs dated 2024 and 2025 are taken at the 2023 level.
CEPCI = {2003:402.0, 2005:468.2, 2006:499.6, 2007:525.4, 2008:575.4, 2010:550.8,
         2014:576.1, 2015:556.8, 2017:567.5, 2018:603.1, 2019:607.5, 2020:596.2,
         2021:708.8, 2022:816.0, 2023:797.9, 2024:797.9, 2025:797.9}
# ECB annual average reference rates, units of currency per USD-equivalent (USD per unit)
FX = {('EUR',2003):1.1312, ('EUR',2014):1.3285, ('EUR',2015):1.1095, ('EUR',2017):1.1297,
      ('EUR',2018):1.1810, ('EUR',2020):1.1422, ('EUR',2021):1.1827, ('EUR',2022):1.0530,
      ('EUR',2023):1.0813, ('EUR',2024):1.0824, ('GBP',2018):1.3355, ('ZAR',2021):1/15.20,
      ('USD',0):1.0}
LHV_H2_kWh = 33.33; HHV_H2_kWh = 39.39     # kWh per kg
RHO_H2 = 0.08988                            # kg per Nm3 at 0 C, 1 atm
KG_PER_SCF = 0.002363

def kgd(v, unit):
    return {'kg/d': v,
            't/d': v*1000,
            'kg/h': v*24,
            'Nm3/h': v*RHO_H2*24,
            'MMSCFD': v*1e6*KG_PER_SCF,
            'MW_HHV': v*1000/HHV_H2_kWh*24,
            'MW_LHV': v*1000/LHV_H2_kWh*24}[unit]

# key, source, feed, tech, ccs, capacity, unit, cost, currency, base_year, scope, note
R = [
 # --- Di Marcoberardino 2018, Processes 6(3) 19 -- island TEC excludes the 22.08 k EUR 700-bar compressor
 ('dimarco18_SRLF','dimarcoberardino2018','biogas','SMR',0,100,'kg/d',(80.20-22.08)*1e3,'EUR',2017,'EQUIP','LFG feed'),
 ('dimarco18_SRAD','dimarcoberardino2018','biogas','SMR',0,100,'kg/d',(74.87-22.08)*1e3,'EUR',2017,'EQUIP','AD feed'),
 ('dimarco18_ATRLF','dimarcoberardino2018','biogas','ATR',0,100,'kg/d',(96.81-22.08)*1e3,'EUR',2017,'EQUIP','LFG feed'),
 ('dimarco18_ATRAD','dimarcoberardino2018','biogas','ATR',0,100,'kg/d',(93.39-22.08)*1e3,'EUR',2017,'EQUIP','AD feed'),
 # --- Di Cicco 2026, Sustainability 18(18) 9330 -- island taken as half of BEC (refuelling line is the other half)
 ('dicicco26_SR200','dicicco2026','biogas','SMR',0,200,'kg/d',0.50*2.22e6,'EUR',2024,'EQUIP','island half of BEC'),
 ('dicicco26_SR500','dicicco2026','biogas','SMR',0,500,'kg/d',0.50*3.97e6,'EUR',2024,'EQUIP','island half of BEC'),
 ('dicicco26_SR1000','dicicco2026','biogas','SMR',0,1000,'kg/d',0.50*6.93e6,'EUR',2024,'EQUIP','island half of BEC'),
 ('dicicco26_ATR200','dicicco2026','biogas','ATR',0,200,'kg/d',0.50*2.61e6,'EUR',2024,'EQUIP','island half of BEC'),
 ('dicicco26_ATR500','dicicco2026','biogas','ATR',0,500,'kg/d',0.50*4.58e6,'EUR',2024,'EQUIP','island half of BEC'),
 ('dicicco26_ATR1000','dicicco2026','biogas','ATR',0,1000,'kg/d',0.50*7.73e6,'EUR',2024,'EQUIP','island half of BEC'),
 # --- Maporti 2022, Chem Eng Trans 96 -- base year taken as 2022, flagged
 ('maporti22_50','maporti2022','biogas','SMR',0,50,'Nm3/h',5.80e5,'EUR',2022,'INST','base year assumed'),
 ('maporti22_100','maporti2022','biogas','SMR',0,100,'Nm3/h',1.00e6,'EUR',2022,'INST','base year assumed'),
 # --- Cormos 2024, Chem Eng Trans 114 -- no-capture case only
 ('cormos24_c1','cormos2024','biogas','SMR',0,100,'MW_LHV',103.79e6,'EUR',2024,'TPC','base year assumed'),
 # --- Wodolazski 2023, Energies 16(17) 6389 -- flagged internally suspect
 ('wodo23','wodolazski2023','biogas','SMR',0,250.67,'kg/h',18.960e6,'EUR',2023,'TPC','suspect, see note'),
 # --- Lee 2021, Applied Sciences 11(13) 6021 -- natural gas
 ('lee21_sc1','lee2021','NG','SMR',0,500,'kg/d',902183,'USD',2015,'TPC',''),
 ('lee21_sc2','lee2021','NG','SMR',0,500,'kg/d',945261,'USD',2015,'TPC','250 C WGS'),
 ('lee21_sc3','lee2021','NG','SMR',0,500,'kg/d',1073098,'USD',2015,'TPC','two WGS'),
 # --- DOE Record 14005
 ('doe14_1500','doe2014','NG','SMR',0,1500,'kg/d',2031647,'USD',2007,'TPC','2005/06 status'),
 ('doe14_1500b','doe2014','NG','SMR',0,1500,'kg/d',1530446,'USD',2007,'TPC','2009/10 status'),
 # --- Yague 2024, Energies 17(5) 1134 -- Table 7 anchors, natural gas
 ('yague_480','yague2024','NG','SMR',0,480,'kg/d',3846130,'USD',2003,'TPC',''),
 ('yague_24k','yague2024','NG','SMR',0,24000,'kg/d',50448185,'USD',2003,'TPC',''),
 ('yague_609k','yague2024','NG','SMR',0,609000,'kg/d',605214704,'USD',2003,'TPC',''),
 # --- GTI 2020 -- SMR baseline, installed
 ('gti20_smr','gti2020','NG','SMR',0,90,'MMSCFD',177950348,'USD',2014,'INST',''),
 # --- IEAGHG 2017-02 base case
 ('ieaghg17_base','ieaghg2017','NG','SMR',0,100000*RHO_H2,'kg/h',170.95e6,'EUR',2014,'TPC','100,000 Nm3/h'),
 # --- NETL 2022-3241 Case 1, no capture
 ('netl22_c1','netl2022','NG','SMR',0,200,'MMSCFD',554*483000,'USD',2018,'TPC',''),
 ('netl22_c1bec','netl2022','NG','SMR',0,200,'MMSCFD',384*483000,'USD',2018,'INST','bare erected cost, includes installation'),
 # --- DOE Liftoff 2024
 ('doe24_smr','doe2024','NG','SMR',0,480,'t/d',360e6,'USD',2022,'INST',''),
 # --- Element Energy 2018 for BEIS -- SMR+CCS cost-capacity curve, GBP/kW HHV
 ('ee18_100','elementenergy2018','NG','SMR',1,100,'MW_HHV',918*100e3,'GBP',2018,'TPC',''),
 ('ee18_200','elementenergy2018','NG','SMR',1,200,'MW_HHV',785*200e3,'GBP',2018,'TPC',''),
 ('ee18_300','elementenergy2018','NG','SMR',1,300,'MW_HHV',700*300e3,'GBP',2018,'TPC',''),
 ('ee18_400','elementenergy2018','NG','SMR',1,400,'MW_HHV',647*400e3,'GBP',2018,'TPC',''),
 ('ee18_500','elementenergy2018','NG','SMR',1,500,'MW_HHV',610*500e3,'GBP',2018,'TPC',''),
 ('ee18_750','elementenergy2018','NG','SMR',1,750,'MW_HHV',550*750e3,'GBP',2018,'TPC',''),
 ('ee18_1000','elementenergy2018','NG','SMR',1,1000,'MW_HHV',529*1000e3,'GBP',2018,'TPC',''),
 # --- with-capture points, kept out of the base fit
 ('sajjadi25_smr','sajjadi2025','NG','SMR',1,100,'t/d',550.55e6,'USD',2025,'TPC',''),
 ('sajjadi25_atr','sajjadi2025','NG','ATR',1,100,'t/d',625.52e6,'USD',2025,'TPC',''),
 ('eluwah23_smr','eluwah2023','NG','SMR',1,2346,'kg/h',183.074e6,'USD',2022,'TPC',''),
 ('netl11_11','netl2011','NG','SMR',1,242,'MMSCFD',492.553e6,'USD',2007,'TPC',''),
 ('netl22_c2','netl2022','NG','SMR',1,200,'MMSCFD',1394*483000,'USD',2018,'TPC',''),
 ('netl22_c3','netl2022','NG','ATR',1,274,'MMSCFD',1092*660000,'USD',2018,'TPC',''),
 ('doe24_smrccs','doe2024','NG','SMR',1,480,'t/d',600e6,'USD',2022,'INST',''),
 ('doe24_atrccs','doe2024','NG','ATR',1,660,'t/d',1100e6,'USD',2022,'INST',''),
 ('ieaghg17_2a','ieaghg2017','NG','SMR',1,100000*RHO_H2,'kg/h',226.07e6,'EUR',2014,'TPC','PSA tail gas MDEA'),
 ('cormos24_c2','cormos2024','biogas','SMR',1,100,'MW_LHV',150.26e6,'EUR',2024,'TPC','MDEA'),
 ('cormos24_c3','cormos2024','biogas','SMR',1,100,'MW_LHV',138.78e6,'EUR',2024,'TPC','membrane'),
 ('gti20_smrccs','gti2020','NG','SMR',1,90,'MMSCFD',317827365,'USD',2014,'INST',''),
]
cols=['id','source','feed','tech','ccs','cap','unit','cost','cur','year','scope','note']
df=pd.DataFrame(R,columns=cols)
df['kg_per_day']=[kgd(c,u) for c,u in zip(df.cap,df.unit)]
if __name__ == "__main__":
    df.to_csv(ROOT / "results" / "capex_raw.csv", index=False)
    print(df.shape, "rows |", df.source.nunique(), "sources |", (df.ccs==0).sum(), "without capture")
