import numpy as np 
from sklearn import preprocessing

example = np.random.randint(1000, size= 1000000)

ohe = preprocessing.OneHotEncoder(sparse_output = False)
ohe_example = ohe.fit_transform(example.reshape(-1,1))
print (f"size of dense array: {ohe_example.nbytes}")

ohe = preprocessing.OneHotEncoder(sparse_output= True)
ohe_example = ohe.fit_transform(example.reshape(-1,1))
print (f"size of sparse array: {ohe_example.data.nbytes}")

full_size = (
    ohe_example.data.nbytes + 
    ohe_example.indptr.nbytes + 
    ohe_example.indices.nbytes
)
print (f"full size of the sparse array: {full_size}")