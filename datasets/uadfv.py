import cv2, torch, glob, random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

class UADFV(Dataset):
    def __init__(self, root, split='train'):
        self.real = glob.glob(f'{root}/real/*/*.mp4')
        self.fake = glob.glob(f'{root}/fake/*/*.mp4')
        self.clips = [(v,0) for v in self.real] + [(v,1) for v in self.fake]
        random.shuffle(self.clips)
        self.tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(),
                             T.Normalize([0.5]*3,[0.5]*3)])
    def __len__(self): return len(self.clips)
    def __getitem__(self, idx):
        vid, label = self.clips[idx]
        cap = cv2.VideoCapture(vid)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, random.randint(0, max(0,frames-1)))
        ret, img = cap.read(); cap.release()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = img[60:220, 80:240]                 # 简易中心裁剪
        return self.tf(Image.fromarray(face)), label