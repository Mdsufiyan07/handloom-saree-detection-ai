import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

DATA_DIR = Path("data/processed")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {DEVICE}")

train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_ds = datasets.ImageFolder(DATA_DIR/"train", transform=train_tf)
val_ds = datasets.ImageFolder(DATA_DIR/"val", transform=val_tf)
print(f"Classes: {train_ds.classes}")
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

best = 0
for epoch in range(10):
    model.train()
    loss_sum = 0
    for x,y in train_loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            _, pred = torch.max(model(x),1)
            total += y.size(0)
            correct += (pred==y).sum().item()
    acc = 100*correct/total
    print(f"Epoch {epoch+1}/10 Loss:{loss_sum/len(train_loader):.3f} Acc:{acc:.1f}%")
    
    if acc > best:
        best = acc
        Path("models").mkdir(exist_ok=True)
        torch.save(model.state_dict(), "models/best_model.pth")
        print(f"--> Saved Best {acc:.1f}%")

print(f"Done! Best Accuracy: {best:.1f}%")