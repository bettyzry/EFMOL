import pandas as pd
import numpy as np
import random

def add_reason(df, num):
    df = df[df.label == 0]
    column_feature = df.columns[:-1]
    data = df.values
    data = data[:,:-1]
    data = abs(data)
    reason = ['0']*len(data)
    y_true = np.zeros(len(data))
    outlier = int(0.2*len(data))

    _avg_ = [np.max(data.T[i]) for i in range(len(column_feature))]
    print(_avg_, len(_avg_))
    for i in range(outlier):
        y_true[i] = 1
        loc = random.randint(0, len(column_feature)-1)
        a = 1 + num
        reason[i] = column_feature[loc]
        data[i][loc] = _avg_[loc] * a
    df = pd.DataFrame(data, columns=column_feature)
    df['label'] = y_true
    df['reason'] = reason
    df = df.sample(frac=1.0)
    df = df.reset_index(drop=True)
    return df

import os
if __name__ == '__main__':
    path = "../data/csv"
    for _, _, files in os.walk(path):  # root 根目录，dirs 子目录
        for filename in files:
            if str(filename)[-4:] == '.csv':
            # if str(filename) == 'cardio.csv':
                filepath = path + "/" + str(filename)
                df = pd.read_csv(filepath, engine='python')
                for i in range(2, 10, 2):
                    path_out = '../data/reason/' + filename[:-4] + '_reason' + str(i/10) + '.csv'
                    reason_df = add_reason(df, i/10)
                    reason_df.to_csv(path_out, index=False)