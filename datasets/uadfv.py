import os
import cv2
import torch
import glob
import random
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image

class UADFV(Dataset):
    def __init__(self, root, split='train'):
        self.root = root
        self.split = split
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.real_videos = glob.glob(os.path.join(root, 'real', '*.mp4'))
        self.fake_videos = glob.glob(os.path.join(root, 'fake', '*.mp4'))
        self.clips = [(v, 0) for v in self.real_videos] + [(v, 1) for v in self.fake_videos]
        random.shuffle(self.clips)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        video_path, label = self.clips[idx]
        cap = cv2.VideoCapture(video_path)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = random.randint(0, frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = self.transform(frame)
        
        return frame, label