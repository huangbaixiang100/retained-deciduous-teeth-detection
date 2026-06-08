# #!/usr/bin/env python3
# """
# ResNet
# :
# -
# -
# """
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, models
# from PIL import Image
# import os
# from pathlib import Path
from config import PROJECT_ROOT

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import numpy as np

# #
# BATCH_SIZE = 16
# NUM_EPOCHS = 50
# LEARNING_RATE = 0.001
# IMAGE_SIZE = 224  # ResNet
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# #
# DATA_DIR = Path('str(PROJECT_ROOT)/cropped_mouth_dataset/cropped_images')
# SAVE_DIR = Path('str(PROJECT_ROOT)/models/classifier')
# SAVE_DIR.mkdir(parents=True, exist_ok=True)

# #
# transform = transforms.Compose([
#     transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# #
# val_transform = transforms.Compose([
#     transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# class OralDiseaseDataset(Dataset):
#     def __init__(self, image_paths, transform=None, augment_minority=False):
#         self.image_paths = image_paths
#         self.transform = transform
#         self.augment_minority = augment_minority

#         #
#         self.other_disease_paths = [p for p in image_paths if p.stem.startswith('other_conditions')]
#         self.normal_paths = [p for p in image_paths if not p.stem.startswith('other_conditions')]

#         #
#         if augment_minority and len(self.other_disease_paths) > 0:
#             #
#             repeat_times = len(self.normal_paths) // len(self.other_disease_paths)
#             #
#             self.other_disease_paths = self.other_disease_paths * repeat_times
#             #
#             self.image_paths = self.other_disease_paths + self.normal_paths

#         #
#         self.strong_transform = transforms.Compose([
#             transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#             transforms.RandomHorizontalFlip(p=0.8),  #
#             transforms.RandomVerticalFlip(p=0.3),    #
#             transforms.RandomRotation(30),           #
#             transforms.RandomAffine(
#                 degrees=0,
#                 translate=(0.2, 0.2),               #
#                 scale=(0.8, 1.2),                   #
#                 shear=15                            #
#             ),
#             transforms.ColorJitter(
#                 brightness=0.4,
#                 contrast=0.4,
#                 saturation=0.4,
#                 hue=0.1
#             ),
#             transforms.RandomPerspective(distortion_scale=0.3, p=0.5),  #
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         image = Image.open(img_path).convert('RGB')
#         label = 1 if img_path.stem.startswith('other_conditions') else 0

#         #
#         if label == 1 and self.augment_minority:
#             image = self.strong_transform(image)
#         elif self.transform:
#             image = self.transform(image)

#         return image, label

# def create_data_loaders():
#     """ (70:10:20)"""
#     #
#     image_paths = list(DATA_DIR.glob('*.jpg')) + list(DATA_DIR.glob('*.png'))

#     #
#     other_disease_paths = [p for p in image_paths if p.stem.startswith('other_conditions')]
#     normal_paths = [p for p in image_paths if not p.stem.startswith('other_conditions')]

#     print(f": {len(other_disease_paths)}")
#     print(f": {len(normal_paths)}")

#     #
#     np.random.seed(42)
#     np.random.shuffle(other_disease_paths)
#     np.random.shuffle(normal_paths)

#     #  (70% , 10% , 20% )
#     #
#     train_idx = int(0.7 * len(other_disease_paths))
#     val_idx = int(0.8 * len(other_disease_paths))
#     train_other = other_disease_paths[:train_idx]
#     val_other = other_disease_paths[train_idx:val_idx]
#     test_other = other_disease_paths[val_idx:]

#     #
#     train_idx = int(0.7 * len(normal_paths))
#     val_idx = int(0.8 * len(normal_paths))
#     train_normal = normal_paths[:train_idx]
#     val_normal = normal_paths[train_idx:val_idx]
#     test_normal = normal_paths[val_idx:]

#     #
#     train_paths = train_other + train_normal
#     val_paths = val_other + val_normal
#     test_paths = test_other + test_normal

#     #
#     np.random.shuffle(train_paths)
#     np.random.shuffle(val_paths)
#     np.random.shuffle(test_paths)

#     print(f"\n:")
#     print(f": {len(train_paths)} ")
#     print(f": {len(val_paths)} ")
#     print(f": {len(test_paths)} ")

#     #
#     train_dataset = OralDiseaseDataset(train_paths, transform, augment_minority=True)
#     val_dataset = OralDiseaseDataset(val_paths, val_transform, augment_minority=False)
#     test_dataset = OralDiseaseDataset(test_paths, val_transform, augment_minority=False)

#     #
#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
#     val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
#     test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

#     return train_loader, val_loader, test_loader

# def train_model():
#     """"""
#     #
#     image_paths = list(DATA_DIR.glob('*.jpg')) + list(DATA_DIR.glob('*.png'))
#     other_disease_paths = [p for p in image_paths if p.stem.startswith('other_conditions')]
#     normal_paths = [p for p in image_paths if not p.stem.startswith('other_conditions')]

#     # ResNet18
#     model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

#     #
#     num_features = model.fc.in_features
#     model.fc = nn.Sequential(
#         nn.Linear(num_features, 256),
#         nn.ReLU(),
#         nn.Dropout(0.5),
#         nn.Linear(256, 2)
#     )

#     model = model.to(DEVICE)

#     #
#     weight_ratio = len(normal_paths) / len(other_disease_paths)
#     class_weights = torch.tensor([1.0, weight_ratio]).to(DEVICE)
#     print(f"\n: [1.0, {weight_ratio:.2f}]")
#     criterion = nn.CrossEntropyLoss(weight=class_weights)
#     optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

#     #
#     train_loader, val_loader, test_loader = create_data_loaders()

#     #
#     train_losses = []
#     val_accuracies = []
#     best_acc = 0.0
#     best_model_state = None

#     print("...")
#     for epoch in range(NUM_EPOCHS):
#         #
#         model.train()
#         total_loss = 0
#         for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}'):
#             images = images.to(DEVICE)
#             labels = labels.to(DEVICE)

#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         avg_loss = total_loss / len(train_loader)
#         train_losses.append(avg_loss)

#         #
#         model.eval()
#         correct = 0
#         total = 0
#         val_loss = 0

#         with torch.no_grad():
#             for images, labels in val_loader:
#                 images = images.to(DEVICE)
#                 labels = labels.to(DEVICE)

#                 outputs = model(images)
#                 _, predicted = torch.max(outputs.data, 1)

#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()

#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item()

#         accuracy = 100 * correct / total
#         val_accuracies.append(accuracy)

#         print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}, Val Accuracy: {accuracy:.2f}%')

#         #
#         if accuracy > best_acc:
#             best_acc = accuracy
#             best_model_state = model.state_dict().copy()
#             print(f': {accuracy:.2f}%')

#         #
#         scheduler.step(accuracy)

#     #
#     plt.figure(figsize=(12, 4))

#     plt.subplot(1, 2, 1)
#     plt.plot(train_losses)
#     plt.title('Training Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')

#     plt.subplot(1, 2, 2)
#     plt.plot(val_accuracies)
#     plt.title('Validation Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy (%)')

#     plt.tight_layout()
#     plt.savefig(SAVE_DIR / 'training_history.png')
#     plt.close()

#     #
#     print("\n...")
#     torch.save({
#         'model_state_dict': best_model_state,
#         'val_accuracy': best_acc,
#         'train_losses': train_losses,
#         'val_accuracies': val_accuracies
#     }, SAVE_DIR / 'best_model.pth')

#     return best_model_state

# def evaluate_model(model_state):
#     """"""
#     #
#     model = models.resnet18(weights=None)
#     num_features = model.fc.in_features
#     model.fc = nn.Sequential(
#         nn.Linear(num_features, 256),
#         nn.ReLU(),
#         nn.Dropout(0.5),
#         nn.Linear(256, 2)
#     )
#     model.load_state_dict(model_state)
#     model = model.to(DEVICE)
#     model.eval()

#     #
#     _, _, test_loader = create_data_loaders()

#     #
#     confusion_matrix = torch.zeros(2, 2)

#     #
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for images, labels in tqdm(test_loader, desc='Evaluating'):
#             images = images.to(DEVICE)
#             labels = labels.to(DEVICE)

#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)

#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#             #
#             for t, p in zip(labels.view(-1), predicted.view(-1)):
#                 confusion_matrix[t.long(), p.long()] += 1

#     #
#     accuracy = 100 * correct / total

#     #
#     precision = confusion_matrix.diag() / confusion_matrix.sum(1)
#     recall = confusion_matrix.diag() / confusion_matrix.sum(0)

#     print('\n:')
#     print(f': {accuracy:.2f}%')
#     print('\n:')
#     print(confusion_matrix)
#     print('\n:')
#     print(':')
#     print(f'  : {precision[0]:.4f}')
#     print(f'  : {recall[0]:.4f}')
#     print(':')
#     print(f'  : {precision[1]:.4f}')
#     print(f'  : {recall[1]:.4f}')

# if __name__ == '__main__':
#     print(f"Device: {DEVICE}")
#     best_model_state = train_model()
#     evaluate_model(best_model_state)
#!/usr/bin/env python3
"""
ResNet
:
-
-
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path
from config import PROJECT_ROOT

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

#
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
IMAGE_SIZE = 224 # ResNet
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#
DATA_DIR = Path('str(PROJECT_ROOT)/cropped_mouth_dataset/cropped_images')
SAVE_DIR = Path('str(PROJECT_ROOT)/models/classifier')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

#
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class OralDiseaseDataset(Dataset):
    def __init__(self, image_paths, transform=None, augment_minority=False):
        self.image_paths = image_paths
        self.transform = transform
        self.augment_minority = augment_minority

        #
        self.other_disease_paths = [p for p in image_paths if p.stem.startswith('other_conditions')]
        self.normal_paths = [p for p in image_paths if not p.stem.startswith('other_conditions')]

        #
        if augment_minority and len(self.other_disease_paths) > 0:
            #
            repeat_times = len(self.normal_paths) // len(self.other_disease_paths)
            #
            self.other_disease_paths = self.other_disease_paths * repeat_times
            #
            self.image_paths = self.other_disease_paths + self.normal_paths

        #
        self.strong_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
 transforms.RandomHorizontalFlip(p=0.8), #
 transforms.RandomVerticalFlip(p=0.3), #
 transforms.RandomRotation(30), #
            transforms.RandomAffine(
                degrees=0,
 translate=(0.2, 0.2), #
 scale=(0.8, 1.2), #
 shear=15 #
            ),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            ),
 transforms.RandomPerspective(distortion_scale=0.3, p=0.5), #
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = 1 if img_path.stem.startswith('other_conditions') else 0

        #
        if label == 1 and self.augment_minority:
            image = self.strong_transform(image)
        elif self.transform:
            image = self.transform(image)

        return image, label

def create_data_loaders(train_paths, val_paths, augment_minority=False):
    """"""
    train_dataset = OralDiseaseDataset(train_paths, transform, augment_minority=augment_minority)
    val_dataset = OralDiseaseDataset(val_paths, val_transform, augment_minority=False)

    #
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, val_loader

def train_model():
    """"""
    #
    image_paths = list(DATA_DIR.glob('*.jpg')) + list(DATA_DIR.glob('*.png'))

    #
    other_disease_paths = [p for p in image_paths if p.stem.startswith('other_conditions')]
    normal_paths = [p for p in image_paths if not p.stem.startswith('other_conditions')]

    #
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []
    best_model_state = None
    best_acc = 0.0

    #
    for fold, (train_idx, val_idx) in enumerate(kfold.split(other_disease_paths + normal_paths)):
        print(f"\nFold {fold+1}")

        #
        train_paths = [image_paths[i] for i in train_idx]
        val_paths = [image_paths[i] for i in val_idx]

        # ResNet18
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        #
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

        model = model.to(DEVICE)

        #
        weight_ratio = len(normal_paths) / len(other_disease_paths)
        class_weights = torch.tensor([1.0, weight_ratio]).to(DEVICE)
        print(f"\n: [1.0, {weight_ratio:.2f}]")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

        #
        train_loader, val_loader = create_data_loaders(train_paths, val_paths, augment_minority=True)

        #
        train_losses = []
        val_accuracies = []

        print("...")
        for epoch in range(NUM_EPOCHS):
            #
            model.train()
            total_loss = 0
            for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}'):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)

            #
            model.eval()
            correct = 0
            total = 0
            val_loss = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)

                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            accuracy = 100 * correct / total
            val_accuracies.append(accuracy)

            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}, Val Accuracy: {accuracy:.2f}%')

            #
            scheduler.step(accuracy)

        #
        fold_acc = np.mean(val_accuracies)
        fold_results.append(fold_acc)

        if fold_acc > best_acc:
            best_acc = fold_acc
            best_model_state = model.state_dict().copy()
            print(f': {fold_acc:.2f}%')

    print(f"\n: {np.mean(fold_results):.2f}%")

    #
    print("\n...")
    torch.save({
        'model_state_dict': best_model_state,
        'val_accuracy': best_acc
    }, SAVE_DIR / 'best_model.pth')

    return best_model_state

def evaluate_model(model_state):
    """"""
    #
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2)
    )
    model.load_state_dict(model_state)
    model = model.to(DEVICE)
    model.eval()

    #
    _, _, test_loader = create_data_loaders()

    #
    confusion_matrix = torch.zeros(2, 2)

    #
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            #
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    #
    accuracy = 100 * correct / total

    #
    precision = confusion_matrix.diag() / confusion_matrix.sum(1)
    recall = confusion_matrix.diag() / confusion_matrix.sum(0)

    print('\n:')
    print(f': {accuracy:.2f}%')
    print('\n:')
    print(confusion_matrix)
    print('\n:')
    print(':')
    print(f'  : {precision[0]:.4f}')
    print(f'  : {recall[0]:.4f}')
    print(':')
    print(f'  : {precision[1]:.4f}')
    print(f'  : {recall[1]:.4f}')

if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    best_model_state = train_model()
    evaluate_model(best_model_state)
