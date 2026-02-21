import pandas as pd


def get_text_series(frame: pd.DataFrame) -> pd.Series:
    """Gets title + description as a data point (Missing vals are replaces w/ empty strings)

    Args:
        frame (pd.DataFrame): dataframe with column title and description

    Returns:
        pd.Series: the concatenated, cleaned data consisting of title and description
    """
    return (
        frame["title"].fillna("") + " " + frame["description"].fillna("")  # type: ignore
    ).str.strip()
