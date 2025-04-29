import pandas as pd
import os
import glob

# Directory path
raw_dir = "./data/raw"

# Find all validated*.tsv files
validated_files = glob.glob(os.path.join(raw_dir, "validated*.tsv"))
print(validated_files)
# Read each file into a DataFrame and store in a list
dfs = [pd.read_csv(file, sep="\t") for file in validated_files]

# Concatenate all DataFrames into one
df = pd.concat(dfs, ignore_index=True)

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the data
total_len = len(df)
train_len = int(total_len * 0.9)
dev_len = int(total_len * 0)

train_df = df[:train_len]
dev_df = df[train_len:train_len + dev_len]
test_df = df[train_len + dev_len:]

# Save the splits to new TSV files
train_df.to_csv(os.path.join(raw_dir, "train.tsv"), sep="\t", index=False)
dev_df.to_csv(os.path.join(raw_dir, "dev.tsv"), sep="\t", index=False)
test_df.to_csv(os.path.join(raw_dir, "test.tsv"), sep="\t", index=False)

print("Dosyalar oluşturuldu: train.tsv, dev.tsv, test.tsv")