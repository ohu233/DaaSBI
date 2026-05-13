import pandas as pd

current_file = 'data/20230917/dataset_20230917_1.csv'
append_file = 'data/20230917/dataset_20230917_2.csv'

df1 = pd.read_csv(current_file, usecols=['index', 'stime', 'lat', 'lon'])
df2 = pd.read_csv(append_file, usecols=['index', 'stime', 'lat', 'lon'])

# 时间转换为纯数字时间戳
df1['stime'] = pd.to_datetime(df1['stime']).astype(int) // 10**9
df2['stime'] = pd.to_datetime(df2['stime']).astype(int) // 10**9


merged_df = pd.concat([df1, df2], ignore_index=True)

merged_df['uid'] = 1
for i in range(1, len(merged_df)):
    if merged_df.loc[i, 'index'] < merged_df.loc[i - 1, 'index']:
        merged_df.loc[i, 'uid'] = merged_df.loc[i - 1, 'uid'] + 1
    else:
        merged_df.loc[i, 'uid'] = merged_df.loc[i - 1, 'uid']

ordered_columns = ['uid', 'index', 'stime', 'lat', 'lon']
merged_df = merged_df[ordered_columns]

merged_df.to_csv('data/dataset_20230917.csv', index=False)

print(merged_df.head(10))