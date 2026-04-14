import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# 1. Load the "Expert Brain"
checkpoint = torch.load("full_disease_model.pth")
class_names = checkpoint['class_names']

model = models.squeezenet1_1()
model.classifier[1] = nn.Conv2d(512, len(class_names), kernel_size=(1,1))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def diagnose(img_path):
    if not os.path.exists(img_path):
        print("Error: Image not found!")
        return

    image = Image.open(img_path).convert('RGB').resize((128, 128))
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    
    # Get the raw result and clean it
    raw_result = class_names[predicted[0]].replace("___", " ").replace("_", " ")
    words = raw_result.split()

    # --- SMART SPLITTING FOR MULTI-WORD PLANTS ---
    if len(words) > 1 and words[0].lower() == "bell" and words[1].lower() == "pepper":
        plant_name = "Bell Pepper"
        disease_name = " ".join(words[2:])
    elif len(words) > 1 and words[0].lower() == "corn" and words[1].lower() == "maize":
        plant_name = "Corn (Maize)"
        disease_name = " ".join(words[2:])
    else:
        plant_name = words[0]
        disease_name = " ".join(words[1:])

    # If no disease words were left, it's a healthy leaf
    if not disease_name.strip():
        disease_name = "Healthy / No disease detected"

    # 4. FINAL CLEAN FORMAT
    print("\n" + "="*40)
    print(f"PLANT   : {plant_name}")
    print(f"DISEASE : {disease_name}")
    print("="*40 + "\n")

if __name__ == "__main__":
    path = input("Enter image filename (e.g. check_this.jpg): ")
    diagnose(path)
