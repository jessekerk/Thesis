import pandas as pd


def get_text_series(frame: pd.DataFrame, use_feature: str = "both") -> pd.Series:
    title = frame["title"].fillna("")
    desc = frame["description"].fillna("")

    if use_feature == "title":
        return title.astype(str).str.strip()
    if use_feature == "description":
        return desc.astype(str).str.strip()

    return (title.astype(str) + " " + desc.astype(str)).str.strip()
