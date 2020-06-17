import os
import pandas as pd
import numpy as np
from pyod.models.iforest import IForest

from six import StringIO
from sklearn import tree
import pydotplus

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

def do_iforest(x, n_estimators=100, max_samples=512):
    clf = IForest(behaviour="new", n_estimators=n_estimators, max_samples=max_samples, random_state=None)
    y_pred = clf.fit_predict(x)
    scores = clf.decision_function(x)
    index = np.where(y_pred == 1)[0]
    return clf, scores, index


def iforest_vis(x, outlier_index, score):
    pca = PCA(n_components=3)  # Reduce to k=3 dimensions
    x = StandardScaler().fit_transform(x)
    x = pca.fit_transform(x)
    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    # Plot the compressed data points
    ax.scatter(x[:, 0], x[:, 1], zs=x[:, 2], s=20, lw=1, c=score, cmap="summer")

    ax2 = fig.add_subplot(212, projection='3d')
    ax2.scatter(x[:, 0], x[:, 1], zs=x[:, 2], s=20, lw=1, label="inliers", c="green")
    ax2.scatter(x[outlier_index, 0], x[outlier_index, 1], x[outlier_index, 2],
               lw=2, s=60, marker="x", c="red", label="outliers")

    plt.show()
    return

# 这里可以调节输出的树的格式，jpg格式必须先输出dot格式
def draw_tree(iForest, tree_list, out_dot=True, out_pdf=True, out_jpg=True):
    for tree_id in tree_list:
        model = iForest.estimators_[tree_id]
        name = 'Tree-' + str(tree_id)
        dot_data = StringIO()
        tree.export_graphviz(model, node_ids=True, filled=True, out_file=dot_data)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

        if out_dot:
            graph.write(name + ".dot")
        if out_pdf:
            graph.write_pdf(name + ".pdf")
        if out_jpg:
            transfer = "dot -Tjpg " + name + ".dot" + " -o " + name + ".jpg"
            os.system(transfer)


def iforest_depth_table(iforest, x):
    n_tree = len(iforest.estimators_samples_)
    n_sample = x.shape[0]
    print("Tree Num: %d" % n_tree)

    depth_table = np.ones([n_sample, n_tree], dtype=int) * -1
    for i in range(n_tree):
        estimator_tree = iforest.estimators_[i].tree_
        n_node = estimator_tree.node_count
        tree_sample_id = iforest.estimators_samples_[i]

        # find data sample id for each node in tree i
        # assign depth for each node in tree i
        print(tree_sample_id)
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


        depth_table[tree_sample_id][i] = 0
        for id in leaf_node_list:
            sample_list = sample_map[id]
            this_depth = node_depth[id]
            depth_table[sample_list, i] = this_depth

    columns = ["Tree-" + str(i) for i in range(n_tree)]
    depth_df = pd.DataFrame(depth_table, dtype=int, index=np.arange(n_sample), columns=columns)
    return depth_df


if __name__ == '__main__':
    file_path = "data/cardio.csv"
    df = pd.read_csv(file_path)
    x = df.values[:, :-1]
    y = df.values[:, -1]
    iforest, if_scores, if_outlier_index = do_iforest(x, n_estimators=100)

    # iforest_vis(x, if_outlier_index, if_scores)
    depth = iforest_depth_table(iforest, x)
    print(depth)
    import pandas as pd

    depth.to_csv('data/d_t.csv')

    n_tree = len(iforest.estimators_samples_)
    draw_tree(iforest, np.arange(10))