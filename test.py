import torch
x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3)
print(x)

x = torch.ones(5, 3)
print(x)


x = torch.eye(5, 3)
print(x)

x = torch.randint(0, 100, (5, 3))
print(x)


X = torch.tensor([[1.0], [2.0], [3.0]])
print(x)

x = torch.scatter(torch.zeros(3,3),1,torch.tensor([[1],[2],[0]]),torch.tensor([[10.0],[20.0],[30.0]]))
print(x)