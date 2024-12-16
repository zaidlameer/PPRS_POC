import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import face_recognition
from torch import nn
from torchvision import models

# Define model architecture
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

# Helper function to convert tensor to image for display
def im_convert(tensor):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inv_normalize = transforms.Normalize(mean=-1 * np.divide(mean, std), std=np.divide([1, 1, 1], std))
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    return np.clip(image, 0, 1)

# Predict function to classify video as real or fake
def predict(model, video_path):
    sequence_length = 20
    im_size = 112
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    frames = []
    vidObj = cv2.VideoCapture(video_path)
    while len(frames) < sequence_length:
        success, frame = vidObj.read()
        if not success:
            break
        faces = face_recognition.face_locations(frame)
        if faces:
            top, right, bottom, left = faces[0]
            frame = frame[top:bottom, left:right, :]
        frame_tensor = train_transforms(frame)
        frames.append(frame_tensor)
    vidObj.release()
    
    frames = torch.stack(frames).unsqueeze(0).cuda()  # Prepare frames as batch tensor
    fmap, logits = model(frames)
    sm = nn.Softmax(dim=1)
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100

    result = "REAL" if prediction.item() == 1 else "FAKE"
    print(f"Prediction: {result} with {confidence:.2f}% confidence.")

# Load model and make prediction
model = Model(num_classes=2).cuda()
model.load_state_dict(torch.load('/content/drive/My Drive/Models/model_87_acc_20_frames_final_data.pt'))
model.eval()

# Path to video for prediction
video_path = "/content/drive/My Drive/DFDC_REAL_Face_only_data/aabqyygbaa.mp4"
predict(model, video_path)
