import torch

# Sample tensor
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
y = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
z = torch.tensor([1, 0, 3])
# Prepare source data of the same shape as the slice (2 elements in this case)
src = torch.tensor([9.0, 8.0])

# --- Tensor operations ---
print("adjoint:\n", x.adjoint())  # Conjugate + transpose

print("arg where:\n", torch.argwhere(torch.tensor([[0, 1], [1, 0]])))

print("cat:\n", torch.cat([x, y], dim=1))

print("concat:\n", torch.concat([x, y], dim=1))

print("concatenate:\n", torch.concatenate([x, y], dim=1))

print("conj:\n", torch.conj(x))

print("chunk:\n", torch.chunk(torch.tensor([1, 2, 3, 4]), 4))

print("dsplit:\n", torch.dsplit(torch.arange(18).reshape(3,2,3),3))

print("column_stack:\n", torch.column_stack((z, z + 2)))

print("dstack:\n", torch.dstack((z, z * 2)))

print("gather:\n", torch.gather(x, 1, torch.tensor([[0, 1], [1, 0]])))

print("h split:\n", torch.hsplit(torch.arange(6).reshape(2, 3), 3))

print("h stack:\n", torch.hstack((z, z + 1)))

print("index_add:\n", torch.index_add(torch.zeros(5), 0, torch.tensor([1, 3, 1]), torch.tensor([9.0, 10.0, 11.0])))

print("index_copy:\n", torch.index_copy(torch.zeros(5), 0, torch.tensor([0, 2]), torch.tensor([5.0, 6.0])))

print("index_reduce:\n", torch.index_reduce(torch.zeros(5), 0, torch.tensor([1, 1, 3]), torch.tensor([2.0, 3.0, 4.0]), reduce="amin"))

print("index_select:\n", torch.index_select(torch.tensor([10, 20, 30, 40]), 0, torch.tensor([1, 3])))

print("masked_select:\n", torch.masked_select(x, x > 2))

print("move dim:\n", torch.movedim(torch.ones(2, 3, 4), 2, 0).shape)

print("move axis:\n", torch.moveaxis(torch.ones(2, 3, 4), 0,1).shape)

print("narrow:\n", torch.narrow(x, 0, 0, 1))

print("narrow_copy:\n", torch.narrow_copy(x, 1, 0, 1))

print("nonzero:\n", torch.nonzero(torch.tensor([[4, 1, 2, 4, 3],[1, 0, 1, 0, 0]])))

print("permute:\n", torch.ones(2, 3, 4).permute(1, 0, 2).shape)

print("reshape:\n", x.reshape(4))

print("row_stack:\n", torch.vstack((z, z ++ 4)))

print("select:\n", x.select(0, 0))

print("scatter:\n", torch.scatter(torch.zeros(2, 4), 1, torch.tensor([[1], [2]]), torch.tensor([[9.0], [10.0]])))

print("diagonal_scatter:\n", torch.diagonal_scatter(torch.zeros(3, 3), torch.tensor([1., 2., 3.])))

print("select_scatter:\n", torch.select_scatter(torch.zeros(2, 2),src, 0,0))

print("slice_scatter:\n", torch.slice_scatter(torch.zeros(3, 3), torch.tensor([[1., 2., 3.]]), dim=0, start=0, end=1))

print("scatter_add:\n", torch.scatter_add(torch.zeros(2, 4), 1, torch.tensor([[0], [1]]), torch.tensor([[9.0], [10.0]])))

print("scatter_reduce:\n", torch.scatter_reduce(torch.zeros(2, 4), 1, torch.tensor([[1], [2]]), torch.tensor([[9.0], [10.0]]), reduce="sum"))

print("split:\n", torch.split(torch.tensor([1, 2, 3, 4]), 4))

print("squeeze:\n", torch.squeeze(torch.tensor([[[1]]])))

print("stack:\n", torch.stack((z, z + 1), dim=0))

print("swap axes:\n", torch.swapaxes(torch.ones(2, 3), 0, 1))

print("swap dims:\n", torch.swapdims(torch.ones(2, 3), 0, 1))

print("t:\n", torch.tensor([[1, 2], [3, 4]]).t())

print("take:\n", torch.take(torch.tensor([[1, 2], [3, 4]]), torch.tensor([0, 3])))

print("take_along_dim:\n", torch.take_along_dim(torch.tensor([[10, 20], [30, 40]]), torch.tensor([[0, 1], [0, 1]]), dim=1))

print("tensor_split:\n", torch.tensor_split(torch.arange(12), 6))

print("tile:\n", torch.tile(torch.tensor([0, 1, 2]), (2,)))

print("transpose:\n", torch.transpose(torch.ones(2, 3), 0, 1))

print("unbind:\n", torch.unbind(torch.tensor([[1, 2], [3, 4]]), dim=0))

print("unravel_index:\n", torch.unravel_index(torch.tensor([2, 5, 7]), (3, 3)))

print("unsqueeze:\n", torch.unsqueeze(torch.tensor([1, 2, 3]), 0))

print("v split:\n", torch.vsplit(torch.arange(6).reshape(3, 2), 3))

print("v stack:\n", torch.vstack((z, z + 1)))

print("where:\n", torch.where(torch.tensor([True, False, True]), torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])))