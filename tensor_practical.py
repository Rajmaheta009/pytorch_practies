import torch
import numpy as np


data = [[1,2],[3,4]]
x2 = torch.tensor([5,6])

#create a dataset with tensor
x1 = torch.tensor(data)
print(x1) #output : tensor([[1, 2],[3, 4]])

#crete dataset with form numpy
array = np.array(data)
x = torch.from_numpy(array)
print(x) #output : tensor([[1, 2],[3, 4]])

# retains the properties of x (tensor data)
x=torch.ones_like(x1)
print(x) #output tensor([[1, 1],[1, 1]])

#override the properties of x (tensor data)
x=torch.rand_like(x1,dtype=torch.float)
print(x) #output tensor([[0.9532, 0.1991],[0.9824, 0.9258]])

#find a shapes
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

#check it data is tensor or not
x = torch.is_tensor(x1)
print(x) #output = true/false

#check it data is get storage or not
x = torch.is_storage(x1)
print(x) #return bool

#check it input data datatype to complex or not
x = torch.is_complex(x1)
print(x) #return bool

#check it input is conjugated or not
x = torch.is_conj(x1)
print(x) #return bool

#check it input datatype is float or not
x = torch.is_floating_point(x1)
print(x) #return bool

# check input => 0
x = torch.is_nonzero(torch.tensor(0))
x1 = torch.tensor([1,2,0])
result = x1 !=0
print(x) #return bool
print(result) #return bool

# set and get both use in torch
torch.set_default_dtype(torch.float64)

a = torch.tensor([1.0, 2.0])
print(a.dtype)  # torch.float64

# Constructs a sparse tensor in COO(rdinate) format with specified values at the given indices.
x = torch.sparse_coo_tensor(x1,x2,size=(10,))
print(x)