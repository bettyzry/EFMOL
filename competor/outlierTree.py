import numpy as np, pandas as pd
from outliertree import OutlierTree
from evaluation import evaluation

def run_outlierTree(df, outliers_print=10):
    outliers_model = OutlierTree()
    outliers_df = outliers_model.fit(df, outliers_print=outliers_print, return_outliers=True)
    results = outliers_df['suspicious_value'].values
    predict = [result['column'] if len(result) > 0 else 'None' for result in results]
    y_pre = pd.DataFrame(predict, columns=['reason'])
    return y_pre

def test():
    path = '../data/reason'
    # out = open('../out/outlierTree.csv', 'w')
    # out.write('name, correct, per\n')
    # out.close()
    for _, _, files in os.walk(path):  # root 根目录，dirs 子目录
        for filename in files:
            if str(filename)[-4:] == '.csv' and str(filename)[:1] == 'c':
                filepath = path + "/" + str(filename)
                df_true = pd.read_csv(filepath)
                df = df_true.drop(['label', 'reason'], axis=1)
                for i in range(10):
                    df_pre = run_outlierTree(df)   # 只有待判断属性列
                    evaluation(df_true, df_pre, outpath='../out/outlierTree', name=filename[:-4])

def show_outlierTree():
    filepath = 'data/reason/cardio_reason0.2.csv'
    df_true = pd.read_csv(filepath)
    df = df_true.drop(['label', 'reason'], axis=1)
    df_pre = run_outlierTree(df)  # 只有待判断属性列
    evaluation(df_true, df_pre, outpath='../out/outlierTree', name='cardio')

import os
if __name__ == '__main__':
    show_outlierTree()

