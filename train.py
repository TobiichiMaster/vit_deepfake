import torch
from torch.utils.data import DataLoader
from datasets.uadfv import UADFV
from model import get_model

def train_model(model, processor, train_dataset, val_dataset, num_epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images = processor(images=images, return_tensors="pt").to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(**images)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = processor(images=images, return_tensors="pt").to(device)
                labels = labels.to(device)
                outputs = model(**images)
                _, predicted = torch.max(outputs.logits, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    model, processor = get_model()
    train_dataset = UADFV(root='data/UADFV', split='train')
    val_dataset = UADFV(root='data/UADFV', split='val')

    train_model(model, processor, train_dataset, val_dataset, num_epochs=5)