import pandas as pd

'''
下载数据后进行预处理，只适用于本地处理
'''
df = pd.read_csv('testdata.csv')

def build_paths_for_group(group_df):
    """为单个 moi_id 组构建路径"""
    group_df = group_df.copy()
    paths = []
    used_indices = set()
    
    # 找到所有起点 (rn_seq=0)
    start_points = group_df[group_df['rn_seq'] == 0]
    
    for _, start_row in start_points.iterrows():
        if start_row.name in used_indices:
            continue
            
        path = []
        current_row = start_row
        used_indices.add(current_row.name)
        path.append(current_row)
        
        # 沿着 next_rn_id -> rn_id 的链条构建路径
        while pd.notna(current_row['next_rn_id']):
            next_candidates = group_df[
                (group_df['rn_id'] == current_row['next_rn_id']) & 
                (group_df['rn_seq'] == current_row['rn_seq'] + 1) &  # 添加 rn_seq 严格 +1 的约束
                (group_df['time'] >= current_row['time']) &  # 添加时间约束：下一节点时间 >= 当前节点时间
                (~group_df.index.isin(used_indices))
            ]
            
            if len(next_candidates) == 0:
                break
                
            # 如果有多个候选，选择第一个
            next_row = next_candidates.iloc[0]
            used_indices.add(next_row.name)
            path.append(next_row)
            current_row = next_row
        
        if len(path) > 0:
            paths.append(pd.DataFrame(path))
    
    return paths

# 按 moi_id 分组并构建路径
all_paths = []
path_id = 0

for moi_id, group in df.groupby('moi_id'):
    paths = build_paths_for_group(group)
    
    for path_df in paths:
        if len(path_df) > 0:
            path_df = path_df.copy()
            path_df['path_id'] = path_id
            # 在每条路径内重新排序
            path_df['path_sequence'] = range(len(path_df))
            all_paths.append(path_df)
            path_id += 1

# 合并所有路径
if all_paths:
    result_df = pd.concat(all_paths, ignore_index=True)
    
    # 按 path_id 和 path_sequence 排序
    result_df = result_df.sort_values(['path_id', 'path_sequence'])
    
    # 如果rn_id和next_rn_id是-1,则删除该行
    result_df = result_df[~((result_df['rn_id'] == -1) & (result_df['next_rn_id'] == -1))]

    # 计算time_diff， 计算方法为当前行的time - 下一行的time，保存在该行的time_diff列
    result_df['next_time'] = result_df.groupby('path_id')['time'].shift(-1)
    result_df['time_diff'] = result_df['next_time'] - result_df['time']

    # 删除 next_rn_id 为 -1 的行
    result_df = result_df[result_df['next_rn_id'] != -1]
    
    # 删除time_diff为空的行
    result_df = result_df[pd.notna(result_df['time_diff'])]
    
    # 若rn_id=-1, 则删除该行，并将下一节点作为新路径的起点，最后重新计算path_id和rn_seq
    def split_paths_by_invalid_nodes(df):
        """处理rn_id=-1的情况，重新分割路径"""
        new_paths = []
        current_path_id = 0
        
        for path_id in df['path_id'].unique():
            path_data = df[df['path_id'] == path_id].copy()
            path_data = path_data.sort_values('path_sequence')
            
            # 找到rn_id=-1的位置
            invalid_indices = path_data[path_data['rn_id'] == -1].index.tolist()
            
            if not invalid_indices:
                # 如果没有无效节点，直接添加整个路径
                path_data['path_id'] = current_path_id
                path_data['rn_seq'] = range(len(path_data))
                new_paths.append(path_data)
                current_path_id += 1
            else:
                # 根据无效节点分割路径
                start_idx = 0
                
                for invalid_idx in invalid_indices:
                    # 获取无效节点在路径中的位置
                    invalid_pos = path_data.index.get_loc(invalid_idx)
                    
                    # 添加无效节点之前的部分作为一个路径
                    if invalid_pos > start_idx:
                        segment = path_data.iloc[start_idx:invalid_pos].copy()
                        segment['path_id'] = current_path_id
                        segment['rn_seq'] = range(len(segment))
                        new_paths.append(segment)
                        current_path_id += 1
                    
                    # 下一段从无效节点后开始
                    start_idx = invalid_pos + 1
                
                # 添加最后一段（如果存在）
                if start_idx < len(path_data):
                    segment = path_data.iloc[start_idx:].copy()
                    segment['path_id'] = current_path_id
                    segment['rn_seq'] = range(len(segment))
                    new_paths.append(segment)
                    current_path_id += 1
        
        if new_paths:
            return pd.concat(new_paths, ignore_index=True)
        else:
            return pd.DataFrame()
    
    # 应用路径分割功能
    result_df = split_paths_by_invalid_nodes(result_df)
    
    # 删除rn_id=-1的行
    result_df = result_df[result_df['rn_id'] != -1]
    
    # 删除只有一个节点的路径（因为需要至少两个节点才能计算time_diff）
    path_sizes = result_df.groupby('path_id').size()
    valid_paths = path_sizes[path_sizes >= 2].index
    result_df = result_df[result_df['path_id'].isin(valid_paths)]
    
    '''    
    # 重新计算path_id（连续编号）
    if len(result_df) > 0:
        unique_paths = result_df['path_id'].unique()
        path_id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_paths)}
        result_df['path_id'] = result_df['path_id'].map(path_id_mapping)
    '''

    
    # 删除 path_sequence 列
    result_df = result_df.drop('path_sequence', axis=1)

    # path_id列移到最前面
    cols = result_df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('path_id')))
    result_df = result_df[cols]

    # 保存结果
    result_df.to_csv('paths_from_DAASBI.csv', index=False)