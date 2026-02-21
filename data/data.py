import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_json("hf://datasets/sh0416/ag_news/train.jsonl", lines=True)

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
dev_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, shuffle=True
)
