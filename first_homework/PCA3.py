# # PCA 降维过程
# 首先加载Iris数据集。
from sklearn import datasets
import numpy as np
from sympy import *

# 算取所有数据进行计算，但是只展现前5行数据

def loadData():
    iris = datasets.load_iris()
    # 只取前五行数据进行计算
    X = iris.data
    y = iris.target[0:5]
    return X

# print("加载iris数据集如下：")
# print(X)
# print()
def PCA(data,n_components):
    # 接下来，需要对数据进行标准化（去均值）。
    # 首先求出每列数据的均值。
    # axis=0 对列求均值
    mean = np.mean(data, axis=0)

    # 去中心化
    # 从每行数据（每次观测中）减去均值：
    data_scaled = data - mean
    # print("去中心化后的数据:")
    # print(X_scaled)
    len_matrix = len(data_scaled)

    # 矩阵转置
    data_scaled = Matrix(data_scaled).T

    # 求协方差矩阵
    cov = data_scaled * Matrix(data_scaled).T
    cov = cov / (len_matrix-1)

    # 特征值、特征向量：
    eigen_vector = [list(line[2][0]) for line in cov.eigenvects()]
    eigen_value = list(cov.eigenvals().keys())

    # 转换成数组,方便得到最大向量组
    eigen_vector = np.array(eigen_vector)
    eigen_value = np.array(eigen_value)

    # 返回的是数组中从小到大排列的值对应的index
    eigen_value_index = np.argsort(eigen_value)
    # 从末尾取二个数，即最大和次最大的特征值在原数组中的下标
    eigen_value_index = eigen_value_index[-1: -(n_components+1): -1]
    # [行 列 步长]  得到最大的特征值对应的最大的特征向量，次最大的特征值对应的次最大的特征向量
    eigen_vector_group = eigen_vector[eigen_value_index, ::]

    res_data = eigen_vector_group * data_scaled
    return res_data


if __name__ == '__main__':
    data = loadData()
    res = PCA(data,2)
    print("原始数据为:")
    print(data)
    print("PCA算法降维后为:")
    # 先矩阵转置  再转换为数组 打印出好看点
    print(np.array(Matrix(res).T)[0:5])






