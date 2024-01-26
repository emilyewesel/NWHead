import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('/dataNAS/people/paschali/datasets/chexpert-public/chexpert-public/train.csv')
print(df.head())