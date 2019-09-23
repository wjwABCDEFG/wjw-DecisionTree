"""
author:wjw
email:975504808@qq.com
create time:2019/09/19 21:55
测试模块，开发过程中测试每个函数，可删勿理
"""
import re
import utils
import numpy as np

# 第一组数据
# 数据集处理
str_raw = '帅。不好。矮。不上进。不嫁。不帅。好。矮。上进。不嫁。帅。好。矮。上进。嫁。不帅。爆好·高。上进。嫁。帅。不好。矮。上进。不嫁。帅。不好。矮。上进。不嫁。帅。好。高。不上进。嫁。不帅。好。中。上进。嫁。帅。爆好·中。上进。嫁。不帅。不好。高。上进。嫁。帅。好。矮。不上进。不嫁。帅。好。矮。不上进。不嫁。'
list1 = re.split(r'[。·]', str_raw)

list2 = []
for row in range(int(len(list1) / 5)):
    list2.append([list1[row * 5 + i] for i in range(5)])

print(list2)
list2 = np.array(list2)
print(type(list2))


print(list2.shape)

datas = list2[:, :-1]
labels = list2[:, -1]
datas_header = ['是否帅', '脾气是否好', '是否高', '是否上进', '结果']
print(datas)
print(labels)
print('=' * 50)


# #第二组数据
# datas = np.array([['1', '1'],
#        ['1', '1'],
#        ['1', '0'],
#        ['0', '1'],
#        ['0', '1']])
#
# labels = np.array(['yes', 'yes', 'no', 'no', 'no'])
# datas_header = ['no-facing', 'flipper']


# # 测试香农熵
# print('测试香农熵')
# testResult = utils.get_shannon_entropy(labels)
# print(testResult)

# #测试条件熵
# print('测试条件熵')
# testResult = utils.get_conditional_entropy(datas[:, 2], labels)
# print(testResult)


# # 测试信息增益
# print('测试信息增益')
# testResult = utils.get_best_gain(datas, labels)
# print(testResult)

# # 测试建树字典
# print(测试建树字典)
# testResult = utils.create_tree(datas_header, datas, labels)
# print(testResult)

# 测试预测函数
# print(测试预测函数)
tree_model = utils.create_tree(datas_header, datas, labels)
input_data = ['帅', '好', '矮', '上进']
testResult = utils.predict_result(tree_model, input_data, datas_header)
print(testResult)

# 测试保存和读取函数
utils.store_tree(tree_model, '嫁不嫁.pkl')     # .pkl和.txt什么的都可以，但建议.pkl
testResult = utils.restore_tree('嫁不嫁.pkl')
print(testResult)
