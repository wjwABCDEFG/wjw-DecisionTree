"""
author:wjw
email:975504808@qq.com
create time:2019/09/19 18:26
"""
import numpy as np


def get_shannon_entropy(labels: np.ndarray) -> float:
    """
    计算香农熵
    即H(X)=−∑(i=1,n)pi * log2pi
    :param labels: shape = (n, )，n个标签
    :return: float香农熵值
    """
    num_labels = len(labels)
    labels_dic = {}     # 考虑到labels不一定是数字，我们用字典来装
    shannon_entropy = 0
    for label in labels:        # 统计各label个数，本例中最后labels_dic格式为{"嫁":6, "不嫁":6}
        if label not in labels_dic.keys():
            labels_dic[label] = 0
        labels_dic[label] += 1

    for i in labels_dic.keys():
        prob = labels_dic[i] / num_labels
        shannon_entropy += - prob * np.log2(prob)

    return shannon_entropy


def get_conditional_entropy(datas: np.ndarray, labels: np.ndarray) -> float:
    """
    接下来的条件熵
    即H(Y|X)=∑(i=1,n)piH(Y|X=xi)     H就是上面的香农熵呀！ Y就是嫁不嫁呀！ X就是当前的特征呀！
    本例中其中一个特征举例：高中矮，则表现为
    H(Y|X) = p矮 * H(嫁|X = 矮) + p中 * H(嫁|X = 中) + p高 * H(嫁|X = 高)
    :param datas:某个特征下所有人的取值，例如['矮', '矮', '矮', '高', '矮', '矮', '高', '中', '中', '高', '矮', '矮']
    :param labels:该特征下嫁或不嫁的结果，例如['不嫁', '不嫁', '嫁', '嫁', '不嫁', '不嫁', '嫁', '嫁', '嫁', '嫁', '不嫁', '不嫁']
    :return:
    """
    num_data = len(datas)
    conditional_entropy = 0
    num_class = {}    # 先得知道身高特征分成了几类（高中矮3类）,每类占多少
    for class_val in datas:     # for循环完后结果为num_class = {"矮": 7, "中": 2, "高": 3}
        if class_val not in num_class.keys():
            num_class[class_val] = 0
        num_class[class_val] += 1

    for i in num_class.keys():
        prob = num_class[i] / num_data
        index = np.argwhere(datas == i).squeeze()  # 以下两句获得datas中'矮'的下标并找出每个'矮'对应的labels值
        # print(index.shape)
        if index.shape != ():          # 这里如果不这么处理会出问题，就是index只有一个数的情况,index = 2，type显示ndarray没有问题，shape却显示()而不是(1,)
            i_labels = labels[index]    # 造成这句的结果居然是i_labels = 'no' 而不是['no'],于是在后面遍历的时候变成了'n'和'o'，这不是我们想要的
        else:
            i_labels = list([])               # 通过先限定i_labels的类型，使用append的方式添加，这时候i_labels就会是想要的['no'],后面遍历的时候就是遍历数组中的这一个元素
            i_labels.append(labels[index])
        conditional_entropy += prob * get_shannon_entropy(i_labels)
    return conditional_entropy


def get_best_gain(datas: np.ndarray, labels: np.ndarray) -> (float, float):
    """
    计算信息增益，并找出最佳的信息增益值，作为当前最佳的划分依据
    即g(D,X)=H(D)−H(D|X)     H就是香农熵呀！ H(D|X)就是信息熵呀！
    best_gain = max(g(D,X))
    :param datas: shape(n, m) n条数据，m种特征
    :param labels: shape = (n, )，n个标签
    :return:返回最佳特征和对应的gain值
    """
    best_gain = 0
    best_feature = -1
    (num_data, num_feature) = datas.shape
    for feature in range(num_feature):
        current_data = datas[:, feature]
        gain = get_shannon_entropy(labels) - get_conditional_entropy(current_data, labels)
        if gain > best_gain:
            best_gain = gain
            best_feature = feature
    return best_feature, best_gain


def create_tree(datas_header: list, datas: np.ndarray, labels: np.ndarray) -> dict:
    """
    get_best_gain()是给一组datas,labels计算一次最佳划分特征
    要构建一棵决策树，那必须要在划分完剩下的数据集继续划分(递归过程)，直到以下情况出现：
    1.剩下全部结果都是相同的，那么直接作为结果。
      如在本例中，假如最佳分组为帅或者不帅，帅对应的labels全部为嫁，则不用继续讨论后面的分组。同理，如果不帅中既有嫁和不嫁，那么需要继续递归
    2.遍历完了所有特征，但是还是无法得到唯一标签，则少数服从多数。
      假如在本例中，遍历到最后一组特征：是否上进，但上进的组里还是有嫁或者不嫁两种标签，且较多为嫁，那就让她嫁
    :param datas_header:
    :param datas:
    :param labels:
    :return:
    """
    # 结束条件1
    if list(labels).count(labels[0]) == len(labels):
        return labels[0]
    # 结束条件2
    if len(datas) == 0 and len(set(labels)) > 1:
        result_num = {}
        for result in labels:
            if result not in result_num.keys():
                result_num[result] = 0
            result_num[result] += 1
        more = -1
        decide = ''
        for result, num in result_num.items():
            if num > more:
                more = num
                decide = result
        return decide

    cur_best_feature_num, _ = get_best_gain(datas, labels)
    cur_best_feature_name = datas_header[cur_best_feature_num]

    # 首先知道该特征下有什么值      本例中class_val = {'帅'， '不帅'}
    class_val = set([data[cur_best_feature_num] for data in datas])
    trees = {cur_best_feature_name: {}}
    for val in class_val:   # 逐一找出每个特征值的数据      本例中表现为含'帅'/'不帅'的数据
        new_datas = [datas[i] for i in range(len(datas)) if datas[i, cur_best_feature_num] == val]    # 用列表生成式，读作:遍历datas每行，找到每行的'是否帅'特征下值为'帅'的行，返回该行
        new_labels = [labels[i] for i in range(len(datas)) if datas[i, cur_best_feature_num] == val]

        new_datas = np.delete(new_datas, cur_best_feature_num, axis=1)     # 删除最佳列，准备进入下一个划分依据
        new_datas_header = np.delete(datas_header, cur_best_feature_num)
        # 递归:去掉该行该列再丢进
        trees[cur_best_feature_name][val] = create_tree(list(new_datas_header), new_datas, np.array(new_labels))
    return trees


def predict_result(trees_model: dict, input_data: np, datas_header: list) -> str:
    """
    1.找字典中的第一个划分特征      本例中是'是否高'
    2.在datas_header中找到高是第几个特征       本例中是第2个特征
    3.在input_data中找到这个特征对应的值        比如要预测的数据中是否高（第二个特征）取值为矮
    4.找到字典中的'矮'的值，如果为str(嫁不嫁)，则直接返回结果，如果是字典，则进行下一个节点预测（递归）
    :param trees_model:
    :param input_data:
    :param datas_header:
    :return:
    """
    cur_judge = list(trees_model.keys())[0]     # '是否高'
    num_feature = datas_header.index(cur_judge)     # 本例中是第2个特征
    cur_val = input_data[num_feature]       # 比如要预测的数据中是否高（第二个特征）取值为矮
    cur_tree = trees_model[cur_judge]
    # print(type(cur_tree[cur_val]))
    if type(cur_tree[cur_val]) == np.str_:
        return cur_tree[cur_val]
    return predict_result(cur_tree[cur_val], input_data, datas_header)


# 保存和读取函数
def store_tree(input_tree, filename):
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(input_tree, f)
        f.close()


def restore_tree(filename):
    import pickle
    with open(filename, 'rb') as f:
        return pickle.load(f)
