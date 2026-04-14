import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# --- 1. SETTINGS ---
DATASET_DIR = "./Dataset" 
LIMIT_PER_CLASS = 200 
EPOCHS = 5

# --- 2. CUSTOM DEEP DATASET ---
class DeepPlantDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = []

        # Find every folder that actually contains images (the diseases)
        for root, dirs, files in os.walk(root_dir):
            images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                folder_name = os.path.basename(root)
                if folder_name.lower() in ['train', 'val', 'test']: # Go one level up if in train/val
                    folder_name = os.path.basename(os.path.dirname(root))
                
                if folder_name not in self.class_names:
                    self.class_names.append(folder_name)
                
                class_idx = self.class_names.index(folder_name)
                for img in images[:LIMIT_PER_CLASS]:
                    self.image_paths.append(os.path.join(root, img))
                    self.labels.append(class_idx)

    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, self.labels[idx]

# --- 3. PREP ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = DeepPlantDataset(DATASET_DIR, transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

print(f"\n--- SUCCESS: FOUND {len(dataset.class_names)} DISEASE CATEGORIES ---")
for i, name in enumerate(dataset.class_names):
    print(f"Slot {i}: {name}")

# --- 4. TRAIN ---
model = models.squeezenet1_1(weights='DEFAULT')
model.classifier[1] = nn.Conv2d(512, len(dataset.class_names), kernel_size=(1,1))
device = torch.device("cpu") # Change to "cuda" if you have a GPU
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    for imgs, lbls in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(imgs.to(device)), lbls.to(device))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Complete.")

torch.save({'model_state_dict': model.state_dict(), 'class_names': dataset.class_names}, "full_disease_model.pth")
