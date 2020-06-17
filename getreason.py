from iForestVis import *
import time
# c_time = str(time.strftime('%Y%m%d %H.%M.%S-', time.localtime(time.time())))
c_time = ''

def zry_iforest_depth_table(iforest, x):
    n_tree = len(iforest.estimators_samples_)
    n_sample = x.shape[0]
    n_feature = x.shape[1]
    print("Tree Num: %d" % n_tree)

    depth_table = np.ones([n_sample, n_tree]) * 0
    score_table = np.ones([n_sample, n_tree, n_feature]) * 0
    for i in range(n_tree):
        estimator_tree = iforest.estimators_[i].tree_
        n_node = estimator_tree.node_count
        tree_sample_id = iforest.estimators_samples_[i]

        # find data sample id for each node in tree i
        # assign depth for each node in tree i
        # print(tree_sample_id)
        sample_map = {0: tree_sample_id}
        node_depth = np.zeros(n_node, dtype=int)
        leaf_node_list = []
        for j in range(n_node):
            # if this node is a leaf node then continue
            if estimator_tree.children_left[j] == -1:
                leaf_node_list.append(j)
                continue
            this_node_sample_id = sample_map[j]
            # get the condition of this node (feature & threshold)
            feature, threshold = estimator_tree.feature[j], estimator_tree.threshold[j]

            left_child, right_child = estimator_tree.children_left[j], estimator_tree.children_right[j]

            node_depth[left_child] = node_depth[j] + 1
            node_depth[right_child] = node_depth[j] + 1
            sample_map[left_child] = this_node_sample_id[np.where(x[this_node_sample_id, feature] <= threshold)[0]]
            sample_map[right_child] = this_node_sample_id[np.where(x[this_node_sample_id, feature] > threshold)[0]]


            left = sample_map[left_child]
            right = sample_map[right_child]
            for k in left:
                score_table[k, i, feature] = 0.5  # 第k个数据，在第i棵树上，第feature个特征的作用
            for k in right:
                score_table[k, i, feature] = 1  # 第k个数据，在第i棵树上，第feature个特征的作用
        # depth_table[tree_sample_id][i] = 0
        for id in leaf_node_list:
            sample_list = sample_map[id]
            this_depth = node_depth[id]
            depth_table[sample_list, i] = 1/this_depth

    columns_tree = ["Tree-" + str(i) for i in range(n_tree)]
    columns_feature = ['Feature-' + str(i) for i in range(n_feature)]
    depth_df = pd.DataFrame(depth_table, index=np.arange(n_sample), columns=columns_tree)
    return depth_df, score_table

def getreason(df):
    x = df.values
    reason_true = x[:, -1]
    y_true = x[:, -2]
    x = x[:, 1:-2]
    n_sample = x.shape[0]
    n_feature = x.shape[1]
    columns_feature = df.columns.tolist()[1:n_feature+1]
    iforest, if_scores, if_outlier_index = do_iforest(x, n_estimators=tree, max_samples=d)
    xt_depth, xtf_depth = zry_iforest_depth_table(iforest, x)
    # depth 归一
    norm_depth = xt_depth.div(xt_depth.sum(axis=1), axis=0)
    norm_depth = norm_depth.fillna(0)
    norm_depth = norm_depth.values
    xf_score = np.ones([n_sample, n_feature]) * 0
    for i in range(n_sample):
        xf_score[i] = np.dot(np.array(norm_depth[i]), np.array(xtf_depth[i]))
    max_index = np.argmax(xf_score, axis=1)
    max_feature = np.array(columns_feature)[max_index]
    index = [i for i in range(len(xf_score))]

    result = pd.DataFrame(xf_score, index=index, columns=columns_feature)
    result['reason'] = max_feature
    return result

def evaluation(df, result, i):
    x = df.values[:, 1:-2]
    n_feature = x.shape[1]
    columns_feature = df.columns.tolist()[1:n_feature]
    reason_true = df['reason'].values
    reason_pre = result['reason'].values
    index = np.where(reason_true != '0')[0]
    reason_true = reason_true[index]
    reason_pre = reason_pre[index]

    count = 0
    # count = np.zeros(len(columns_feature))
    for j in range(len(reason_true)):
        if reason_pre[j] == reason_true[j]:
            count += 1
    # print(count, len(index), count/len(index))
    out = open('out/all' + str(i) + '.csv', 'a')
    out.write(str(tree) + ','+ str(d) + ',' + str(count) + ',' + str(count/len(index)) + '\n')
    out.close()

    # from sklearn.metrics import confusion_matrix  # 生成混淆矩阵函数
    # cm = confusion_matrix(reason_true, reason_pre)
    # print(cm)
    # plot_confusion_matrix(cm, columns_feature)
    # print(columns_feature)


def plot_confusion_matrix(cm, labels_name, title='Confusion Matrix'):
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    # plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    # plt.title(title)    # 图像标题
    # plt.colorbar()
    # num_local = np.array(range(len(labels_name)))
    # plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    # plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    ax = sns.heatmap(cm, annot=True)
    plt.title('N=' + str(d))  # 图像标题
    plt.savefig('out/confusionMatrix' + message + '.png')
    # plt.show()
    plt.close()
    return

import seaborn as sns
import time
from max_n_index import *
if __name__ == '__main__':
    tree_list = [50, 100, 150, 200, 250, 300]
    # tree_list = [300]
    d_list = [256, 512, 1024]

    i = 0
    path = 'D:/0学习/0数据集/多维/cardio_reason.csv'
    df = pd.read_csv(path, engine='python')
    for i in range(10):
        out = open('out/all' + str(i) +'.csv', 'w')
        out.write('tree, d, correct, per\n')
        out.close()
        for d in d_list:
            for tree in tree_list:
                message = str(tree) + '-' + str(d)
                result = getreason(df)
                # result.to_csv('out/result' + str(c_time) + message + '.csv')
                evaluation(df, result, i)