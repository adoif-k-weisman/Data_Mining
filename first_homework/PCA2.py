# # PCA 降维过程
# 首先加载Iris数据集。
from sklearn import datasets
import numpy as np
from sympy import *

# 这是份保留注释和调试代码的文件，用于不备之需


def loadData():
    iris = datasets.load_iris()
    # 只取前五行数据进行计算
    # X = iris.data[0:5]
    X = iris.data
    y = iris.target[0:5]
    return X


def PCA(data, n_components):
    # 接下来，需要对数据进行标准化（去均值）。
    # 首先求出每列数据的均值。

    # axis=0 对列求均值
    mean = np.mean(data, axis=0)

    # 去中心化
    # 从每行数据（每次观测中）减去均值：
    data_scaled = data - mean
    # print("len=={}".format(len(data_scaled)))
    # print(data_scaled[0:5])

    len_matrix = len(data_scaled)
    # 矩阵转置
    data_scaled = Matrix(data_scaled).T

    # 求协方差矩阵
    cov = data_scaled * Matrix(data_scaled).T
    # print("data_scaled:")
    # print(data_scaled)
    # print("len:")
    # print(len(data_scaled))
    cov = cov / (len_matrix-1)
    # print(np.array(cov))
    # 特征值、特征向量：
    eigen_vector = [list(line[2][0]) for line in cov.eigenvects()]
    eigen_value = list(cov.eigenvals().keys())

    # 转换成数组,方便得到最大向量组
    eigen_vector = np.array(eigen_vector)
    eigen_value = np.array(eigen_value)
    print(eigen_vector)

    # 返回的是数组中从小到大排列的值对应的index
    eigen_value_index = np.argsort(eigen_value)
    # 从末尾取二个数，即最大和次最大的特征值在原数组中的下标
    eigen_value_index = eigen_value_index[-1: -3: -1]

    # 行 列 步长
    redEigVects = eigen_vector[eigen_value_index, ::]
    # print(redEigVects)
    res_data = redEigVects * data_scaled
    return res_data


# 将特征向量按特征值排序：
# 获取特征值按降序排序对应原矩阵的下标
# idx = eigen_value.argsort()[::-1]
# #idx = np.argsort(eigen_value)
# eigen_value = eigen_value[idx]
# eigen_vector = eigen_vector[:,idx]

# 降维：去掉对我们计算影响量不大的数据
# 选择前两个特征值：
if __name__ == '__main__':
    data = loadData()
    print("原始数据为:")
    # print(data)
    print("长度")
    print(len(data))
    res = PCA(data, 2)
    print("PCA算法降维后为:")
    print(res)






