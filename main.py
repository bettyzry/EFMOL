from getreason import getreason
import pandas as pd

if __name__ == '__main__':
    path = "data/cardio.csv"
    df = pd.read_csv(path)
    getreason(df)