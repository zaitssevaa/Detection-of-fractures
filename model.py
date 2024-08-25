# model.py
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 4)  # Пример для регрессии на ограничивающие рамки

    def forward(self, x):
        return self.model(x)

def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for images, targets in dataloader:
            outputs = model(images)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

def main():
    # Примерный код для создания фейковых данных
    class DummyDataset(Dataset):
        def __init__(self, size):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            # Возвращает случайное изображение и метку
            return torch.randn(3, 224, 224), torch.tensor([0.0, 0.0, 0.0, 0.0])

    # Инициализируем модель, критерион и оптимизатор
    model = MyModel()
    dataset = DummyDataset(size=100)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Обучаем модель
    train_model(model, dataloader, criterion, optimizer)

    # Сохраняем модель
    save_model(model, 'model.pth')

if __name__ == "__main__":
    main()