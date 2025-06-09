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

# --- Basic Tensors ---
t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print("tensor:\n", t)

# --- Sparse Tensors ---
x1 = torch.tensor([[0, 1], [1, 2]])
x2 = torch.tensor([3.0, 4.0])
sparse_coo = torch.sparse_coo_tensor(x1, x2, size=(3, 3))
print("sparse_coo_tensor:\n", sparse_coo)

data = [1, 2, 3, 4]
crow_indices = torch.tensor([0, 2, 4])
col_indices = torch.tensor([0, 1, 1, 2])
sparse_csr = torch.sparse_csr_tensor(crow_indices, col_indices, torch.tensor(data))
print("sparse_csr_tensor:\n", sparse_csr)

ccol_indices = torch.tensor([0, 2, 4])
row_indices = torch.tensor([0, 1, 1, 2])
sparse_csc = torch.sparse_csc_tensor(ccol_indices, row_indices, torch.tensor(data))
print("sparse_csc_tensor:\n", sparse_csc)

block_data = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
bsr = torch.sparse_bsr_tensor(crow_indices, col_indices, block_data.view(2, 2, 2))
print("sparse_bsr_tensor:\n", bsr)

bsc = torch.sparse_bsc_tensor(ccol_indices, row_indices, block_data.view(2, 2, 2))
print("sparse_bsc_tensor:\n", bsc)

# --- Conversions ---
arr = [1, 2, 3]
asarr = torch.asarray(arr)
print("asarray:\n", asarr)

astens = torch.as_tensor(arr)
print("as_tensor:\n", astens)

x = torch.tensor([1, 2, 3, 4])
as_strided_view = torch.as_strided(x, size=(2,), stride=(2,))
print("as_strided:\n", as_strided_view)

# from_numpy
np_array = np.array([[1, 2], [3, 4]])
from_np = torch.from_numpy(np_array)
print("from_numpy:\n", from_np)

# frombuffer
buffer = bytearray([1, 0, 0, 0, 2, 0, 0, 0])
from_buf = torch.frombuffer(buffer, dtype=torch.int32)
print("frombuffer:\n", from_buf)

# from_dlpack (mock example, normally from another lib)
# Not executed here, would require real DLPack data
# from_dlpack = torch.from_dlpack(dlpack_tensor)

# --- Initialization ---
print("zeros:\n", torch.zeros(2, 3))
print("zeros_like:\n", torch.zeros_like(t))

print("ones:\n", torch.ones(2, 3))
print("ones_like:\n", torch.ones_like(t))

print("arange:\n", torch.arange(0, 10, 2))
print("range:\n", torch.range(0, 5, 1))  # deprecated in favor of arange

print("linspace:\n", torch.linspace(0, 1, steps=5))
print("logspace:\n", torch.logspace(1, 2, steps=5, base=10.0))

print("eye:\n", torch.eye(3))
print("empty:\n", torch.empty(2, 2))
print("empty_like:\n", torch.empty_like(t))
print("empty_strided:\n", torch.empty_strided((2, 3), (1, 2)))

print("full:\n", torch.full((2, 2), 7))
print("full_like:\n", torch.full_like(t, 9))

# --- Quantized ---
float_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
q_tensor = torch.quantize_per_tensor(float_tensor, scale=0.1, zero_point=10, dtype=torch.quint8)
print("quantize_per_tensor:\n", q_tensor)

per_channel_q = torch.quantize_per_channel(float_tensor, scales=torch.tensor([0.1, 0.2]), zero_points=torch.tensor([0, 0]), axis=0, dtype=torch.qint8)
print("quantize_per_channel:\n", per_channel_q)

dequant = q_tensor.dequantize()
print("dequantize:\n", dequant)

# --- Complex ---
real = torch.tensor([1.0, 2.0])
imag = torch.tensor([3.0, 4.0])
complex_tensor = torch.complex(real, imag)
print("complex:\n", complex_tensor)

polar_tensor = torch.polar(torch.tensor([1.0, 2.0]), torch.tensor([0.0, 3.14]))
print("polar:\n", polar_tensor)

# --- Heaviside ---
x = torch.tensor([-1.5, 0, 2.0])
y = torch.tensor([1.0, 0.5, 0.0])
print("heaviside:\n", torch.heaviside(x, y))
