"""Base layers for the maps: municipal polygons and departments, WGS84, simplified.

Writes results/mun.gpkg, results/dept.gpkg and results/mun_points.csv (a point inside each municipal polygon, used only where the DANE seat coordinates are missing).

Run from the project root:
    python scripts/06_geo.py
"""
import pathlib
import warnings

import geopandas as gpd

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

g = gpd.read_file(ROOT / "inputs" / "municipal_boundary_polygon.shp")
g = g.to_crs(4326)
g["MpCodigo"] = g["MpCodigo"].astype(str).str.zfill(5)
g["dept_code"] = g["MpCodigo"].str[:2]
mun = g[["MpCodigo", "MpNombre", "dept_code", "geometry"]].copy()
mun["geometry"] = mun.geometry.simplify(0.005)
mun.to_file(RES / "mun.gpkg", driver="GPKG")
dept = g[["dept_code", "geometry"]].dissolve(by="dept_code").reset_index()
dept["geometry"] = dept.geometry.simplify(0.01)
dept.to_file(RES / "dept.gpkg", driver="GPKG")
pts = g.to_crs(3116).representative_point().to_crs(4326)
out = g[["MpCodigo", "MpNombre"]].copy()
out["lon"], out["lat"] = pts.x.round(5), pts.y.round(5)
out.to_csv(RES / "mun_points.csv", index=False)
print(f"{len(mun)} municipalities, {len(dept)} departments written")
