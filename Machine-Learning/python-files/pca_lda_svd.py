from sklearn.decomposition import TruncatedSVD
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

# PRINCIPAL COMPONENT ANALYSIS
pca = PCA(n_components=2)
pca.fit(data)
pca_transformed_data = pca.transform(data)

print("Original data shape:", data.shape)
print("Transformed data shape:", pca_transformed_data.shape)
print("Transformed data:", pca_transformed_data)


# LINEAR DISCRIMIMNANT ANALYSIS
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(data, [1, 0, 1])
lda_transformed_data = lda.transform(data)

print("Original data shape:", data.shape)
print("Transformed data shape:", lda_transformed_data.shape)
print("Transformed data:", lda_transformed_data)


# SINGULAR VALUE DECOMPOSITION
U, s, V = np.linalg.svd(data)

print("Original data shape:", data.shape)
print("Singular values:", s)
print("Left singular vectors (U):", U)
print("Right singular vectors (V):", V)


# SVD ALTERNATE
tsvd = TruncatedSVD(n_components=2)
tsvd.fit(data)
transformed_svd = tsvd.transform(data)
print(transformed_svd)
