"""
Сверточная нейронная сеть (CNN) для распознавания рукописных цифр MNIST
Замена MLP на CNN по замечанию преподавателя
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("🧠 СВЕРТОЧНАЯ НЕЙРОННАЯ СЕТЬ (CNN) ДЛЯ MNIST")
print("=" * 70)

# ============================================
# 1. ПОДГОТОВКА ДАННЫХ
# ============================================
print("\n📊 1. ЗАГРУЗКА ДАННЫХ MNIST")
print("-" * 50)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

print(f"   ✅ Обучающая выборка: {len(train_dataset)} изображений")
print(f"   ✅ Тестовая выборка: {len(test_dataset)} изображений")

# Разделение на train/validation
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_data, val_data = random_split(train_dataset, [train_size, val_size])

print(f"   📊 Train: {len(train_data)} | Validation: {len(val_data)} | Test: {len(test_dataset)}")

batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ============================================
# 2. СОЗДАНИЕ МОДЕЛИ CNN
# ============================================
print("\n🧠 2. АРХИТЕКТУРА СВЕРТОЧНОЙ НЕЙРОННОЙ СЕТИ (CNN)")
print("-" * 50)

class DigitClassifierCNN(nn.Module):
    """
    Сверточная нейронная сеть для MNIST
    Архитектура: Conv -> Conv -> FC
    """
    def __init__(self):
        super(DigitClassifierCNN, self).__init__()
        # Первый сверточный слой (вход: 1 канал, выход: 32 канала)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Второй сверточный слой (вход: 32, выход: 64)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Полносвязный слой
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Conv1 -> ReLU -> MaxPool
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Conv2 -> ReLU -> MaxPool
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Преобразуем в вектор
        x = x.view(-1, 64 * 7 * 7)

        # Полносвязные слои
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

model = DigitClassifierCNN()
print("   ✅ Сверточная модель CNN создана!")
print("   📊 Архитектура:")
print("      Conv1: 1 -> 32 (3x3)")
print("      MaxPool (2x2)")
print("      Conv2: 32 -> 64 (3x3)")
print("      MaxPool (2x2)")
print("      FC1: 3136 -> 128")
print("      FC2: 128 -> 10")

# ============================================
# 3. НАСТРОЙКА ОБУЧЕНИЯ
# ============================================
print("\n⚙️ 3. НАСТРОЙКА ОБУЧЕНИЯ")
print("-" * 50)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

print(f"   📉 Функция потерь: CrossEntropyLoss")
print(f"   🎯 Оптимизатор: Adam")
print(f"   🔁 Эпох: {num_epochs}")

# ============================================
# 4. ОБУЧЕНИЕ
# ============================================
print("\n🚀 4. ОБУЧЕНИЕ МОДЕЛИ")
print("-" * 50)

train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    # Обучение
    model.train()
    total_train_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Валидация
    model.eval()
    total_val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    accuracy = 100 * correct / total
    val_accuracies.append(accuracy)

    print(f"   Эпоха [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {accuracy:.2f}%")

print("\n   ✅ Обучение завершено!")

# ============================================
# 5. ГРАФИКИ
# ============================================
print("\n📈 5. СОЗДАНИЕ ГРАФИКОВ")
print("-" * 50)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(range(1, num_epochs+1), train_losses, 'b-', label='Train Loss', linewidth=2)
ax1.plot(range(1, num_epochs+1), val_losses, 'r-', label='Validation Loss', linewidth=2)
ax1.set_xlabel('Эпоха')
ax1.set_ylabel('Потери (Loss)')
ax1.set_title('График обучения: Функция потерь (CNN)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, num_epochs+1), val_accuracies, 'g-', marker='o', linewidth=2)
ax2.set_xlabel('Эпоха')
ax2.set_ylabel('Точность (%)')
ax2.set_title('График обучения: Точность на валидации (CNN)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cnn_training_plots.png', dpi=150, bbox_inches='tight')
print("   ✅ Графики сохранены как 'cnn_training_plots.png'")

# ============================================
# 6. ОЦЕНКА НА ТЕСТЕ
# ============================================
print("\n🎯 6. ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ")
print("-" * 50)

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"   📊 Точность на тесте (CNN): {test_accuracy:.2f}%")

# ============================================
# 7. ПРИМЕРЫ ПРЕДСКАЗАНИЙ
# ============================================
print("\n🖼️ 7. ПРИМЕРЫ ПРЕДСКАЗАНИЙ")
print("-" * 50)

test_loader_vis = DataLoader(test_dataset, batch_size=16, shuffle=True)
images, labels = next(iter(test_loader_vis))

with torch.no_grad():
    outputs = model(images)
    _, predictions = torch.max(outputs, 1)

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
fig.suptitle('CNN: Примеры распознавания рукописных цифр', fontsize=14, fontweight='bold')

for i, ax in enumerate(axes.flat):
    img = images[i].squeeze().numpy()
    ax.imshow(img, cmap='gray')
    color = 'green' if predictions[i] == labels[i] else 'red'
    ax.set_title(f'Реальная: {labels[i]} | Предсказано: {predictions[i]}', color=color, fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig('cnn_predictions_examples.png', dpi=150, bbox_inches='tight')
print("   ✅ Примеры сохранены как 'cnn_predictions_examples.png'")

# ============================================
# 8. ИТОГОВЫЙ ОТЧЕТ
# ============================================
print("\n" + "=" * 70)
print("📊 ИТОГОВЫЙ ОТЧЕТ (CNN)")
print("=" * 70)
print(f"""
✅ ПРОЕКТ ВЫПОЛНЕН УСПЕШНО!

📋 ЗАДАЧА:
   • Классификация рукописных цифр (MNIST)
   • 10 классов (цифры 0-9)

📊 ДАННЫЕ:
   • Train: {len(train_data)} изображений
   • Validation: {len(val_data)} изображений
   • Test: {len(test_dataset)} изображений

🧠 МОДЕЛЬ (CNN):
   • Архитектура: Conv(1→32) → Conv(32→64) → FC(3136→128) → FC(128→10)
   • Функция активации: ReLU
   • Оптимизатор: Adam
   • MaxPool: 2x2 после каждой свертки
   • Dropout: 0.25

📈 РЕЗУЛЬТАТЫ:
   • Точность на валидации: {val_accuracies[-1]:.2f}%
   • Точность на тесте (CNN): {test_accuracy:.2f}%

💡 ВЫВОД:
   Сверточная сеть (CNN) показывает лучший результат, чем MLP.
   CNN учитывает пространственную структуру изображений.
""")
print("=" * 70)
print("🎉 ГОТОВО! CNN модель обучена!")
print("=" * 70)