import pandas as pd
import numpy as np

# ============================================================
# 阈值
# ============================================================
STAY_MIN_DUR = 600
STAY_KEEP_INTERVAL = 3600
MOVE_MIN_DT = 180
MOVE_MIN_HEX = 8        # full Manhattan |dx|+|dy|+|dz|, 每格移动=2, 4=至少2格

# ============================================================
# 输入
# ============================================================
INPUT_FILE = "data\dataset_multicity_with_hex.csv"
df = pd.read_csv(INPUT_FILE)

# 修正 hex_z 确保 x+y+z=0
mask_bad = (df["hex_x"] + df["hex_y"] + df["hex_z"]) != 0
n_bad = mask_bad.sum()
if n_bad > 0:
    print(f"修正 hex_z: {n_bad} 行不满足 x+y+z=0")
    df.loc[mask_bad, "hex_z"] = -df.loc[mask_bad, "hex_x"] - df.loc[mask_bad, "hex_y"]

print(f"原始: {len(df):,} 行, {df['uid'].nunique()} 用户")

# ============================================================
# 降采样（按天+用户分组）
# ============================================================
df["_date"] = pd.to_datetime(df["stime"], unit="s").dt.strftime("%Y%m%d")
df["_group"] = df["_date"] + "_" + df["uid"].astype(str)
results = []

for gid, group in df.groupby("_group"):
    g = group.sort_values("stime").reset_index(drop=True)

    # ---- 阶段1: 压缩同 hex 停留 ----
    g["_hex_key"] = g["hex_x"].astype(str) + "_" + g["hex_y"].astype(str)
    g["_run_id"] = (g["_hex_key"] != g["_hex_key"].shift()).cumsum()

    phase1 = set()
    for _, run in g.groupby("_run_id"):
        indices = list(run.index)
        first, last = indices[0], indices[-1]
        dur = g.at[last, "stime"] - g.at[first, "stime"]
        if dur < STAY_MIN_DUR:
            phase1.update(indices)
        else:
            phase1.add(first)
            phase1.add(last)
            for h in range(1, int(dur / STAY_KEEP_INTERVAL)):
                target_t = h * STAY_KEEP_INTERVAL
                for idx in indices:
                    if g.at[idx, "stime"] - g.at[first, "stime"] >= target_t:
                        phase1.add(idx)
                        break

    # ---- 阶段2: 稀疏化移动 ----
    p1_sorted = sorted(phase1)
    kept = [p1_sorted[0]]
    for idx in p1_sorted[1:]:
        last = kept[-1]
        dt = g.at[idx, "stime"] - g.at[last, "stime"]
        dh = (abs(g.at[idx, "hex_x"] - g.at[last, "hex_x"])
              + abs(g.at[idx, "hex_y"] - g.at[last, "hex_y"])
              + abs(g.at[idx, "hex_z"] - g.at[last, "hex_z"]))
        if dt >= MOVE_MIN_DT or dh >= MOVE_MIN_HEX:
            kept.append(idx)
    if kept[-1] != p1_sorted[-1]:
        kept.append(p1_sorted[-1])

    # ---- 累积 time_value / dist_value ----
    for i, idx in enumerate(kept):
        row = g.iloc[idx].to_dict()
        prev = kept[i - 1] if i > 0 else -1
        tv = g.loc[prev + 1 : idx, "time_value"].sum()
        dv = g.loc[prev + 1 : idx, "dist_value"].sum()
        row["time_value"] = int(tv)
        row["dist_value"] = dv
        row["velocity"] = dv / tv * 3.6 if tv > 0 else 0.0
        row["idx"] = i + 1
        results.append(row)

out = pd.DataFrame(results)
out = out.sort_values(["uid", "stime"]).reset_index(drop=True)

# 清理临时列，保留原始列序
drop_cols = ["_date", "_group", "_hex_key", "_run_id"]
out = out.drop(columns=[c for c in drop_cols if c in out.columns])
cols = [c for c in df.columns if c not in drop_cols]
out = out[cols]

print(f"降采样后: {len(out):,} 行 (保留 {len(out)/len(df)*100:.1f}%)")

# ============================================================
# OD 统计 & 输出
# ============================================================
DISTANCE_THRESHOLD = 36   # full Manhattan, 每格=2, 36=18格

od_count = 0
for uid, grp in out.groupby("uid"):
    g = grp.sort_values("stime")
    anchor = g.iloc[0]
    for i in range(1, len(g)):
        cur = g.iloc[i]
        mh = abs(cur["hex_x"] - anchor["hex_x"]) + abs(cur["hex_y"] - anchor["hex_y"]) + abs(cur["hex_z"] - anchor["hex_z"])
        if mh >= DISTANCE_THRESHOLD:
            od_count += 1
            anchor = cur

fname = "data\\dataset_multicity_with_hex_downsampled.csv"
out.to_csv(fname, index=False, encoding="utf-8-sig")
print(f"  {fname}: {len(out)} 行, {od_count} OD 对")
