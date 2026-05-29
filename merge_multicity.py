import os
import pandas as pd

def merge_multicity_files(input_folder, output_file):
    # List to hold dataframes
    dataframes = []

    # Iterate through files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)
            df = pd.read_csv(file_path)
            dataframes.append(df)

    # Concatenate all dataframes into one
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Save the merged dataframe to a new CSV file
    merged_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_folder = 'data/multicity_dataset'  # Update this path
    output_file = 'data/multicity_dataset/dataset_multicity.csv'  # Update this path
    merge_multicity_files(input_folder, output_file)