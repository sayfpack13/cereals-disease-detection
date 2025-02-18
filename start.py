import argparse
import torch
from torchvision import models,datasets
from torchvision import transforms
from PIL import Image
import torch.nn as nn

class CONFIG:
    IMAGE_HEIGHT = 224
    IMAGE_WIDTH = 224
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DROPOUT = 0.3
    DATASET_PATH = "dataset"


transform = transforms.Compose([
    transforms.Resize((CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    
def load_model(checkpoint_path, num_classes):
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=CONFIG.DROPOUT, inplace=True),
        nn.Linear(model.classifier[1].in_features, num_classes, bias=True)
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=CONFIG.DEVICE)
    model.load_state_dict(checkpoint)
    model.to(CONFIG.DEVICE)
    model.eval()
    
    return model

def predict(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return predicted.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on an image.")
    parser.add_argument("--model", required=True, help="Path to the model file")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes in the model")
    args = parser.parse_args()


    dataset = datasets.ImageFolder(root=CONFIG.DATASET_PATH, transform=transform)
    class_names = dataset.classes
    model = load_model(args.model, num_classes=len(class_names))

    prediction = predict(model, args.image)
    print(class_names[prediction])