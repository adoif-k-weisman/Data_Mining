# 用sklearn进行PCA
# 降维后的数据（前五行）为：
from sklearn import datasets

iris = datasets.load_iris()
# 只取前五行数据进行计算
X = iris.data


from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(X)
lowDmat=pca.transform(X)#降维后的数据
print(lowDmat[0:5])