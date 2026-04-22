import pandas as pd

df = pd.read_csv('paths_from_DAASBI_20.csv')

# 用path_id分组
grouped_df = df.groupby('path_id')

result_rows = []

# 在每个组内处理
for name, group in grouped_df:
    group = group.copy().reset_index(drop=True)
    
    if len(group) == 0:
        continue
    
    result = []
    cumulative_time_diff = 0
    cumulative_distance = 0
    start_idx = 0  # 记录累计计算的起始索引
    
    for i in range(len(group)):
        cumulative_time_diff += group.iloc[i]['time_diff']
        cumulative_distance += group.iloc[i]['distance']
        
        # 当累计time_diff超过1000时，保留起始行并更新累计值
        if cumulative_time_diff > 1000:
            # 保留区间的第一行（start_idx），并将累计值保存到这一行
            segment_row = group.iloc[start_idx].copy()
            segment_row['time_diff'] = cumulative_time_diff
            segment_row['distance'] = cumulative_distance
            
            result.append(segment_row)
            
            # 重置累计值，更新起始索引为下一行
            cumulative_time_diff = 0
            cumulative_distance = 0
            start_idx = i + 1  # 下一段从i+1开始
    
    # 处理最后剩余的数据
    if start_idx < len(group):
        # 计算从start_idx到最后的累计值
        segment_time = sum(group.iloc[j]['time_diff'] for j in range(start_idx, len(group)))
        segment_distance = sum(group.iloc[j]['distance'] for j in range(start_idx, len(group)))
        
        # 保留区间的第一行
        last_row = group.iloc[start_idx].copy()
        last_row['time_diff'] = segment_time
        last_row['distance'] = segment_distance
        result.append(last_row)
    
    # 处理轨迹首尾对应关系
    if len(result) > 1:
        for j in range(len(result) - 1):
            # 当前行的next信息 = 下一行的当前信息
            result[j]['next_rn_id'] = result[j+1]['rn_id']
            result[j]['next_loc_x'] = result[j+1]['loc_x']
            result[j]['next_loc_y'] = result[j+1]['loc_y']
            
            # 如果还有经纬度信息，也要对应
            if 'next_lat' in result[j] and 'lat' in result[j+1]:
                result[j]['next_lat'] = result[j+1]['lat']
            if 'next_lon' in result[j] and 'lon' in result[j+1]:
                result[j]['next_lon'] = result[j+1]['lon']
    
    # 将当前组的结果转换为DataFrame并添加到结果列表
    if result:
        group_result = pd.DataFrame(result)
        result_rows.append(group_result)

# 合并所有结果
if result_rows:
    result_df = pd.concat(result_rows, ignore_index=True)
    
    # 计算速度：距离除以时间差
    # 避免除零错误，当time_diff为0时，velocity设为0
    result_df['velocity'] = result_df.apply(
        lambda row: row['distance'] / row['time_diff'] if row['time_diff'] > 0 else 0, 
        axis=1
    )
    
    # 保存结果
    result_df.to_csv('processed.csv', index=False)