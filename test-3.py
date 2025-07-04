import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt

# 1. Generate binary classification data
X, y = make_classification(n_samples=500, n_features=2, n_classes=2,
                           n_informative=2, n_redundant=0, random_state=42)
X = X.astype('float32')
y = y.astype('float32').reshape(-1, 1)  # Reshape for BCELoss

# 2. Train-test split + normalization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Convert to tensors
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)

# 4. Define Logistic Regression model
class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)  # 2 input features, 1 output

    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # sigmoid gives probability

model = LogisticRegressionModel()

# 5. Define Loss and Optimizer
loss_fn = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 6. Train the Model
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# 7. Evaluate
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test)
    predictions = (y_pred_test > 0.5).float()
    accuracy = (predictions == y_test).float().mean().item()
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
