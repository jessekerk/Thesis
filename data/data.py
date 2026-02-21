# data/data.py
import pandas as pd
from sklearn.model_selection import train_test_split
from data.pollution import DataPolluter

df = pd.read_json("hf://datasets/sh0416/ag_news/train.jsonl", lines=True)

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
dev_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, shuffle=True
)

# Example: pollute ONLY "description" for 20% of rows
polluter = DataPolluter(seed=42)
train_polluted = polluter.pollute(
    train_df, feature="description", pollution_rate=0.2, mode="empty"
).df
dev_polluted = polluter.pollute(
    dev_df, feature="description", pollution_rate=0.2, mode="empty"
).df
test_polluted = polluter.pollute(
    test_df, feature="description", pollution_rate=0.2, mode="empty"
).df
