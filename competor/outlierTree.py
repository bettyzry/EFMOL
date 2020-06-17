import numpy as np, pandas as pd
from outliertree import OutlierTree
from evaluation import evaluation

def run_outlierTree(df, outliers_print=10):
    outliers_model = OutlierTree()
    outliers_df = outliers_model.fit(df, outliers_print=outliers_print, return_outliers=True)
    print(outliers_df)
    results = outliers_df['suspicious_value'].values
    predict = [result['column'] if len(result) > 0 else 'None' for result in results]
    y_pre = pd.DataFrame(predict, columns=['reason'])
    return y_pre


if __name__ == '__main__':
    df1 = pd.read_csv('../data/cardio_reason.csv',
                     usecols=['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14',
                              'A15', 'A16', 'A17', 'A18', 'A19', 'A20'])
    df2 = pd.read_csv('../data/cardio_reason.csv')
    outliers_printlist = [10, 20, 40, 80, 160]
    for i in range(10):
        out = open('../out/all' + str(i) + '.csv', 'w')
        out.write('outliers_print, d, correct, per\n')
        out.close()
        for outliers_print in outliers_printlist:
            y_pre = run_outlierTree(df1)
            evaluation(df2, y_pre, i, tree=outliers_print)

