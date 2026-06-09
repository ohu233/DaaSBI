import os
import pandas as pd

def merge_multicity_files(input_folder, output_file):
    dataframes = []

    # 读取所有文件（原始文件不改动）
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)
            df = pd.read_csv(file_path)
            # 标记来源文件，用于区分不同文件中相同的 uid
            df['_src'] = filename
            dataframes.append(df)

    # 合并所有数据
    merged_df = pd.concat(dataframes, ignore_index=True)

    # 只在输出中重新编码 uid：按（来源文件 + 原始uid）组合，从 1 开始依次编号
    if 'uid' in merged_df.columns:
        merged_df['uid'] = pd.factorize(merged_df['_src'].astype(str) + '_' + merged_df['uid'].astype(str))[0] + 1
        merged_df.drop(columns=['_src'], inplace=True)

    merged_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_folder = 'data/multicity_dataset'
    output_file = 'data/multicity_dataset/dataset_multicity.csv'
    merge_multicity_files(input_folder, output_file)
