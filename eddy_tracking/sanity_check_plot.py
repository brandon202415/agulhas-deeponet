import datetime, tempfile, os, sys
import numpy as np
import netCDF4
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRED_NPZ = "/Users/brandonzhang/Downloads/best/predictions.npz"
NLAT, NLON = 61, 101
NSENS = NLAT * NLON
VARIABLES = ["zos", "uo", "vo", "thetao", "so", "mlotst"]
DAY = 50

d = np.load(PRED_NPZ)
lon = d["lon"].astype("float64")
lat = d["lat"].astype("float64")
y_true, y_pred, y_persist = d["y_true"], d["y_pred"], d["y_persist"]

def block(arr, day, vname):
    vi = VARIABLES.index(vname)
    col = arr[day, vi*NSENS:(vi+1)*NSENS]
    return col.reshape(NLAT, NLON)

zos_all_true = np.stack([block(y_true, t, "zos") for t in range(y_true.shape[0])])
ocean_mask = zos_all_true.std(axis=0) > 1e-4

def get_fields(arr, day):
    zos = np.where(ocean_mask, block(arr, day, "zos"), np.nan).astype("float32")
    uo = np.where(ocean_mask, block(arr, day, "uo"), np.nan).astype("float32")
    vo = np.where(ocean_mask, block(arr, day, "vo"), np.nan).astype("float32")
    return zos, uo, vo

def write_nc(path, zos, uo, vo, lon, lat):
    with netCDF4.Dataset(path, "w") as h:
        h.createDimension("lat", len(lat)); h.createDimension("lon", len(lon))
        vlon = h.createVariable("lon", "f8", ("lon",)); vlon[:] = lon; vlon.units = "degrees_east"
        vlat = h.createVariable("lat", "f8", ("lat",)); vlat[:] = lat; vlat.units = "degrees_north"
        vadt = h.createVariable("adt", "f4", ("lat", "lon"), fill_value=np.float32(np.nan)); vadt[:] = zos; vadt.units = "m"
        vu = h.createVariable("u", "f4", ("lat", "lon"), fill_value=np.float32(np.nan)); vu[:] = uo; vu.units = "m/s"
        vv = h.createVariable("v", "f4", ("lat", "lon"), fill_value=np.float32(np.nan)); vv[:] = vo; vv.units = "m/s"

def identify(zos, uo, vo, day_idx):
    from py_eddy_tracker.dataset.grid import RegularGridDataset
    tmp = tempfile.mktemp(suffix=".nc")
    try:
        write_nc(tmp, zos, uo, vo, lon, lat)
        h = RegularGridDataset(tmp, "lon", "lat", nan_masking=True)
        h.bessel_high_filter("adt", 400)
        date = datetime.datetime(2000, 1, 1) + datetime.timedelta(days=int(day_idx))
        anti, cyclo = h.eddy_identification("adt", "u", "v", date, step=0.005, shape_error=70, pixel_limit=(1, 2000))
        return anti, cyclo
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

zos_t, uo_t, vo_t = get_fields(y_true, DAY)
zos_p, uo_p, vo_p = get_fields(y_pred, DAY)
zos_s, uo_s, vo_s = get_fields(y_persist, DAY)

anti_t, cyc_t = identify(zos_t, uo_t, vo_t, DAY)
anti_p, cyc_p = identify(zos_p, uo_p, vo_p, DAY)
anti_s, cyc_s = identify(zos_s, uo_s, vo_s, DAY)

print(f"Day {DAY}: true anti={len(anti_t)} cyc={len(cyc_t)} | model anti={len(anti_p)} cyc={len(cyc_p)} | persist anti={len(anti_s)} cyc={len(cyc_s)}")
print("\nTrue anticyclonic centers (lon, lat):")
for i in range(len(anti_t)):
    print(f"  #{i}: ({anti_t['lon'][i]:.4f}, {anti_t['lat'][i]:.4f})")
print("Model anticyclonic centers (lon, lat):")
for i in range(len(anti_p)):
    print(f"  #{i}: ({anti_p['lon'][i]:.4f}, {anti_p['lat'][i]:.4f})")
print("Persist anticyclonic centers (lon, lat):")
for i in range(len(anti_s)):
    print(f"  #{i}: ({anti_s['lon'][i]:.4f}, {anti_s['lat'][i]:.4f})")
print("\nTrue cyclonic centers (lon, lat):")
for i in range(len(cyc_t)):
    print(f"  #{i}: ({cyc_t['lon'][i]:.4f}, {cyc_t['lat'][i]:.4f})")
print("Model cyclonic centers (lon, lat):")
for i in range(len(cyc_p)):
    print(f"  #{i}: ({cyc_p['lon'][i]:.4f}, {cyc_p['lat'][i]:.4f})")
print("Persist cyclonic centers (lon, lat):")
for i in range(len(cyc_s)):
    print(f"  #{i}: ({cyc_s['lon'][i]:.4f}, {cyc_s['lat'][i]:.4f})")

# Plot: SSH (true) as background, all three sets of centers marked
fig, ax = plt.subplots(figsize=(9, 7))
pc = ax.pcolormesh(lon, lat, zos_t, shading="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
plt.colorbar(pc, ax=ax, label="SSH (m), true field")
if len(anti_t): ax.scatter(anti_t["lon"], anti_t["lat"], marker="*", s=250, c="black", label="true (anticyclonic)", zorder=5, edgecolors="white")
if len(cyc_t): ax.scatter(cyc_t["lon"], cyc_t["lat"], marker="*", s=250, c="black", zorder=5, edgecolors="white")
if len(anti_p): ax.scatter(anti_p["lon"], anti_p["lat"], marker="x", s=150, c="red", label="model", zorder=6)
if len(cyc_p): ax.scatter(cyc_p["lon"], cyc_p["lat"], marker="x", s=150, c="red", zorder=6)
if len(anti_s): ax.scatter(anti_s["lon"], anti_s["lat"], marker="+", s=200, c="blue", label="persistence", zorder=6)
if len(cyc_s): ax.scatter(cyc_s["lon"], cyc_s["lat"], marker="+", s=200, c="blue", zorder=6)
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title(f"Day {DAY}: detected eddy centers, true SSH background\n(grid spacing ~48-55 km)")
ax.legend(loc="upper right")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("/Users/brandonzhang/Downloads/eddy/eddy_tracking/sanity_check/day50_map.png", dpi=150)
print("\nSaved map to eddy_tracking/sanity_check/day50_map.png")
