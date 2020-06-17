import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # 绘图库
from getreason import getreason


def evaluation(df, result, times, tree=0, d=0):
    x = df.values[:, 1:-2]
    n_feature = x.shape[1]
    columns_feature = df.columns.tolist()[1:n_feature]
    reason_true = df['reason'].values
    reason_pre = result['reason'].values
    index = np.where(reason_true != '0')[0]
    reason_true = reason_true[index]
    reason_pre = reason_pre[index]

    count = 0
    for j in range(len(reason_true)):
        if reason_pre[j] == reason_true[j]:
            count += 1
    out = open('../out/all' + str(times) + '.csv', 'a')
    out.write(str(tree) + ','+ str(d) + ',' + str(count) + ',' + str(count/len(index)) + '\n')
    out.close()
    from sklearn.metrics import confusion_matrix  # 生成混淆矩阵函数

    print(reason_true, reason_pre)
    cm = confusion_matrix(reason_true, reason_pre)
    plot_confusion_matrix(cm, title='Confusion Matrix_' + str(tree), message=str(tree))


def plot_confusion_matrix(cm, title='Confusion Matrix', message=''):
    ax = sns.heatmap(cm, annot=True)
    plt.title(title)  # 图像标题
    plt.savefig('../out/confusionMatrix' + message + '.png')
    plt.close()
    return
