import pandas as pd

current_file = 'data/20230917/dataset_multicity_20230917_processed_1.csv'
append_file = 'data/20230917/dataset_multicity_20230917_processed_2.csv'

df1 = pd.read_csv(current_file)
df2 = pd.read_csv(append_file)

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

# uid放在最前面
cols = merged_df.columns.tolist()
cols.insert(0, cols.pop(cols.index('uid')))
merged_df = merged_df[cols]

merged_df.to_csv('data/dataset_multicity_20230917_processed.csv', index=False)

print(merged_df.head(10))