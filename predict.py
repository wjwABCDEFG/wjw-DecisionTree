"""
author:wjw
email:975504808@qq.com
create time:2019/09/23 2:20
训练完后就可以用决策树模型来预测了
"""
import utils

# 从文件中加载已经训练好的模型
print('加载模型...')
tree_model = utils.restore_tree('嫁不嫁.pkl')

# 要预测的数据样本
datas_header = ['是否帅', '脾气是否好', '是否高', '是否上进', '结果']
input_data = ['帅', '好', '矮', '上进']

# 打印预测结果
result = utils.predict_result(tree_model, input_data, datas_header)
print('预测结果为：' + result)
