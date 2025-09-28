import torch
import torchreid
from torchvision import transforms
import cv2
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------------- ReID Model ----------------
model = torchreid.models.build_model(
    name='osnet_ain_x1_0',
    num_classes=1000,
    pretrained=True
)
model.eval()

# preprocessing for TorchReID
preprocess = transforms.Compose([
    transforms.Resize((256, 128)),  # height x width
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

people_database = []

def get_embedding(img_bgr):
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    img_tensor = preprocess(img_pil).unsqueeze(0)  # add batch dim
    with torch.no_grad():
        features = model(img_tensor)  # shape [1, feature_dim]
    return features.squeeze(0).numpy()  # convert to numpy vector

def get_person_id(person_clip):
    found_person = False
    best_match = 0
    best_match_id = -1
    for i in range(len(people_database)):
        match_value = cosine_similarity(people_database[i].reshape(1, -1), person_clip.reshape(1, -1))
        if match_value > 0.8:
            found_person = True
            if match_value > best_match:
                best_match = match_value
                best_match_id = i
    # if we don't find it we add it
    if not found_person and best_match < 0.25:
        people_database.append(person_clip)
        return len(people_database) - 1
    else:
        return best_match_id
