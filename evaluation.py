import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # 绘图库


def evaluation(df_true, df_pre, outpath='out/result', name='None'):
    df_true = df_true[:len(df_pre)]
    x = df_true.values[:, :-2]
    n_feature = x.shape[1]
    columns_feature = df_true.columns.tolist()[:n_feature]
    reason_true = df_true['reason'].values
    reason_pre = df_pre['reason'].values
    index = np.where(reason_true != '0')[0]
    reason_true = reason_true[index]
    reason_pre = reason_pre[index]

    count = 0
    for j in range(len(reason_true)):
        if reason_pre[j] == reason_true[j]:
            count += 1
    out = open(outpath + '_' + name + '.csv', 'a')

    out.write(name + ',' + str(count) + ',' + str(count/len(index)) + '\n')
    out.close()
    from sklearn.metrics import confusion_matrix  # 生成混淆矩阵函数

    cm = confusion_matrix(reason_true, reason_pre)
    plot_confusion_matrix(cm, outpath = outpath + '_' + name)


def plot_confusion_matrix(cm, title='Confusion Matrix', outpath='out/ConfusionMatrix'):
    ax = sns.heatmap(cm, annot=True)
    plt.title(title)  # 图像标题
    plt.savefig(outpath + '.png')
    plt.close()
    return
