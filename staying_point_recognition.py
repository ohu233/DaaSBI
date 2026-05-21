import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def recognize_staying_points(
    df,
    uid=None,
    uid_col="uid",
    time_col="time_value",
    dist_col="dist_value",
    speed_threshold=1.0,          # D/T < 1 m/s 判为停驻，对应论文 D < T
    adjust_speed_threshold=0.5,   # 论文 label adjustment 中提到的 0.5
    min_stay_time=300,            # 小于 5 分钟的候选停驻改为移动
    max_prev_jump=400,            # 与前一簇跳变距离过大，候选停驻改为移动
    filter_speed_kmh=120          # 清洗漂移点，速度超过 120km/h 删除
):
    data = df.copy()

    if uid is not None:
        data[uid_col] = data[uid_col].astype(str)
        data = data[data[uid_col] == str(uid)].copy()

    data[time_col] = pd.to_numeric(data[time_col], errors="coerce")
    data[dist_col] = pd.to_numeric(data[dist_col], errors="coerce")
    data = data.dropna(subset=[time_col, dist_col]).copy()

    # 保留原始顺序；如果你有真实时间戳列，可以先按时间戳排序
    data = data.reset_index(drop=True)

    # 基础清洗：删除负值、异常高速漂移点
    data = data[(data[time_col] >= 0) & (data[dist_col] >= 0)].copy()

    speed = data[dist_col] / data[time_col].replace(0, np.nan)
    data["instant_speed_mps"] = speed.fillna(0)

    if filter_speed_kmh is not None:
        max_speed_mps = filter_speed_kmh / 3.6
        data = data[data["instant_speed_mps"] <= max_speed_mps].copy()

    data = data.reset_index(drop=True)

    if len(data) < 2:
        raise ValueError("数据点太少，无法识别停驻点。")

    # 构造 2D 轨迹：累计时间、累计距离
    data["cumulative_time"] = data[time_col].cumsum()
    data["cumulative_dist"] = data[dist_col].cumsum()

    # 论文中的 base spatiotemporal cost: S0 = Dmean * Tmean / 2
    d_mean = data[dist_col].mean()
    t_mean = data[time_col].mean()
    S0 = d_mean * t_mean / 2

    if S0 <= 0 or np.isnan(S0):
        S0 = 1e-9

    def area_of_indices(indices):
        """窗口面积：D * T"""
        part = data.loc[indices]
        D = part["cumulative_dist"].max() - part["cumulative_dist"].min()
        T = part["cumulative_time"].max() - part["cumulative_time"].min()
        return D * T

    def metrics_of_indices(indices):
        part = data.loc[indices]
        start_time = part["cumulative_time"].iloc[0]
        end_time = part["cumulative_time"].iloc[-1]
        duration = end_time - start_time

        dist_range = part["cumulative_dist"].max() - part["cumulative_dist"].min()
        avg_speed = dist_range / duration if duration > 0 else 0

        return {
            "start_idx": int(indices[0]),
            "end_idx": int(indices[-1]),
            "start_time": float(start_time),
            "end_time": float(end_time),
            "duration": float(duration),
            "dist_range": float(dist_range),
            "avg_speed": float(avg_speed)
        }

    # ---------- Algorithm 1: Trajectory Clustering ----------
    clusters = []
    start = 0
    n = len(data)

    for j in range(1, n):
        idxs = list(range(start, j + 1))
        area = area_of_indices(idxs)
        k = len(idxs)

        # 新加入点使窗口面积超过 k * S0，则前一个窗口成簇，当前点作为新窗口起点
        if area > k * S0:
            if j - 1 >= start:
                clusters.append(list(range(start, j)))
            start = j

    if start < n:
        clusters.append(list(range(start, n)))

    # ---------- Algorithm 2: Window Dividing ----------
    def divide_cluster(indices):
        if len(indices) <= 3:
            return [indices]

        scores = []
        # prefix 长度从 2 到 len(indices)
        for k in range(2, len(indices) + 1):
            prefix = indices[:k]
            scores.append(area_of_indices(prefix) - k * S0)

        split_pos = None
        for i in range(len(scores) - 1):
            # 差值序列开始增加，认为到达分割拐点
            if scores[i] < scores[i + 1]:
                split_pos = i + 2
                break

        if split_pos is None or split_pos <= 1 or split_pos >= len(indices):
            return [indices]

        left = indices[:split_pos]
        right = indices[split_pos:]

        return divide_cluster(left) + divide_cluster(right)

    divided_clusters = []
    for c in clusters:
        divided_clusters.extend(divide_cluster(c))

    # ---------- Algorithm 3: Label Assignment ----------
    labeled = []
    for c in divided_clusters:
        m = metrics_of_indices(c)
        # D/T < speed_threshold 判为停驻
        label = 1 if m["avg_speed"] < speed_threshold else 0
        m["label"] = label
        m["indices"] = c
        labeled.append(m)

    # ---------- Window Merging: 合并相邻同标签窗口 ----------
    merged = []
    for item in labeled:
        if not merged or merged[-1]["label"] != item["label"]:
            merged.append({
                "label": item["label"],
                "indices": item["indices"].copy()
            })
        else:
            merged[-1]["indices"].extend(item["indices"])

    # 合并后重新计算 D/T 和标签
    relabeled = []
    for m in merged:
        stat = metrics_of_indices(m["indices"])
        stat["label"] = 1 if stat["avg_speed"] < speed_threshold else 0
        stat["indices"] = m["indices"]
        relabeled.append(stat)

    # ---------- Label Adjustment ----------
    adjusted = []
    for i, item in enumerate(relabeled):
        label = item["label"]

        if label == 1:
            # 规则1：短于 5 分钟，不认为是真停驻
            if item["duration"] < min_stay_time:
                label = 0

            # 规则2：候选停驻自身速度仍偏高，改为移动
            if item["avg_speed"] > adjust_speed_threshold:
                label = 0

            # 规则3：与前一簇的距离跳变过大，可能是移动片段误判
            if i > 0:
                prev_end_idx = relabeled[i - 1]["end_idx"]
                cur_start_idx = item["start_idx"]
                jump_dist = abs(
                    data.loc[cur_start_idx, "cumulative_dist"]
                    - data.loc[prev_end_idx, "cumulative_dist"]
                )
                if jump_dist > max_prev_jump:
                    label = 0

        item = item.copy()
        item["label"] = label
        adjusted.append(item)

    # 再合并一次，避免 adjustment 后出现相邻移动窗口
    final_clusters = []
    for item in adjusted:
        if not final_clusters or final_clusters[-1]["label"] != item["label"]:
            final_clusters.append({
                "label": item["label"],
                "indices": item["indices"].copy()
            })
        else:
            final_clusters[-1]["indices"].extend(item["indices"])

    final_stats = []
    for cluster_id, item in enumerate(final_clusters):
        stat = metrics_of_indices(item["indices"])
        stat["cluster_id"] = cluster_id
        stat["label"] = item["label"]
        stat["state"] = "stay" if item["label"] == 1 else "move"
        final_stats.append(stat)

    cluster_df = pd.DataFrame(final_stats)

    # 给原始点打标签
    data["cluster_id"] = -1
    data["stay_label"] = 0
    data["state"] = "move"

    for _, row in cluster_df.iterrows():
        idxs = list(range(int(row["start_idx"]), int(row["end_idx"]) + 1))
        data.loc[idxs, "cluster_id"] = int(row["cluster_id"])
        data.loc[idxs, "stay_label"] = int(row["label"])
        data.loc[idxs, "state"] = row["state"]

    stay_points = cluster_df[cluster_df["label"] == 1].copy()

    return data, cluster_df, stay_points, S0


# ===================== 使用示例 =====================

df = pd.read_csv(r"data\dataset_20230917_nanjing_to_gaochun_lishui.csv")
df["uid"] = df["uid"].astype(str)

point_df, cluster_df, stay_points, S0 = recognize_staying_points(
    df,
    uid="1",
    uid_col="uid",
    time_col="time_value",
    dist_col="dist_value"
)

print("S0 =", S0)
print("\n所有轨迹簇：")
print(cluster_df)

print("\n识别出的停驻点：")
print(stay_points)

# 保存结果
point_df.to_csv("uid_1_stay_point_labels.csv", index=False, encoding="utf-8-sig")
stay_points.to_csv("uid_1_stay_points_summary.csv", index=False, encoding="utf-8-sig")


# ===================== 可视化 =====================

plt.figure(figsize=(12, 6))
plt.plot(
    point_df["cumulative_time"],
    point_df["cumulative_dist"],
    linewidth=1.8,
    label="trajectory"
)

# 用阴影标出停驻区间
for _, row in stay_points.iterrows():
    plt.axvspan(
        row["start_time"],
        row["end_time"],
        alpha=0.25
    )

# 标出停驻点中心
if len(stay_points) > 0:
    stay_mid_time = (stay_points["start_time"] + stay_points["end_time"]) / 2
    stay_mid_dist = []
    for _, row in stay_points.iterrows():
        part = point_df.iloc[int(row["start_idx"]): int(row["end_idx"]) + 1]
        stay_mid_dist.append(part["cumulative_dist"].mean())

    plt.scatter(
        stay_mid_time,
        stay_mid_dist,
        marker="o",
        s=60,
        label="recognized stay"
    )

plt.xlabel("Cumulative Time")
plt.ylabel("Cumulative Distance")
plt.title("Adaptive Staying Point Recognition")
plt.legend()
plt.show()