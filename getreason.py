from iForestVis import *
from evaluation import evaluation
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

def getreason(df, n_estimators=300, max_samples=1024):
    x = df.values
    n_sample = x.shape[0]
    n_feature = x.shape[1]
    columns_feature = df.columns.tolist()[:n_feature]

    chunk = 3000
    final_result = pd.DataFrame()
    for j in range(0, len(df), chunk):
        data = x[j: min(j+chunk, len(df))]
        n_sample = data.shape[0]
        iforest, if_scores, if_outlier_index = do_iforest(data, n_estimators=300, max_samples=1024)
        xt_depth, xtf_depth = zry_iforest_depth_table(iforest, data)
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
        final_result = final_result.append(result)
    final_result = final_result.reset_index(drop=True)
    print(len(final_result))
    return final_result


def test():
    path = 'data/reason'
    # out = open('out/2EFMOL.csv', 'w')
    # out.write('name, correct, per\n')
    # out.close()
    for _, _, files in os.walk(path):  # root 根目录，dirs 子目录
        for filename in files:
            # if str(filename)[-4:] == '.csv':
            if str(filename)[-4:] == '.csv' and str(filename)[:1] == 'c':
                filepath = path + "/" + str(filename)
                df_true = pd.read_csv(filepath)
                df = df_true.drop(['label', 'reason'], axis=1)
                for i in range(10):
                    df_pre = getreason(df)   # 只有待判断属性列
                    evaluation(df_true, df_pre, outpath='out/2EFMOL', name=filename[:-4])


def show_efmol():
    n_estimators = 300
    max_samples = 1024
    dataset = 'cardio_reason0.2'

    filepath = 'data/reason/' + dataset + '.csv'
    df_true = pd.read_csv(filepath)
    df = df_true.drop(['label', 'reason'], axis=1)
    df_pre = getreason(df, n_estimators, max_samples)  # 只有待判断属性列
    evaluation(df_true, df_pre, outpath='out/EFMOL', name=dataset)


if __name__ == '__main__':
    show_efmol()


