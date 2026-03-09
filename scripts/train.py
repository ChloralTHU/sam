import os
import random
import torch
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


# ==============================
# Paths
# ==============================

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_IMG = os.path.join(ROOT, "dataset/img")
DATA_MASK = os.path.join(ROOT, "dataset/mask")

MODEL_PATH = os.path.join(ROOT, "models/sam_vit_b.pth")

CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


device = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================
# Hyperparameters
# ==============================

EPOCHS = 30
LR = 2e-5
POS_INTERNAL = 3
POINT_JITTER = 4


# ==============================
# Dataset
# ==============================

class SEMDataset(Dataset):

    def __init__(self,img_dir,mask_dir):

        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.files = sorted([
            f for f in os.listdir(img_dir)
            if f.endswith(".tif") or f.endswith(".tiff")
        ])

        print("dataset size:",len(self.files))

        self.transform = ResizeLongestSide(1024)

        self.pixel_mean = torch.tensor([123.675,116.28,103.53]).view(3,1,1)
        self.pixel_std  = torch.tensor([58.395,57.12,57.375]).view(3,1,1)

    def read_img(self,path):

        img = cv2.imread(path,cv2.IMREAD_UNCHANGED)

        if img.dtype == np.uint16:
            img = (img/256).astype(np.uint8)

        if len(img.shape)==2:
            img = np.stack([img]*3,-1)

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        return img

    def read_mask(self,path):

        mask = cv2.imread(path,cv2.IMREAD_UNCHANGED)

        if len(mask.shape)==3:
            mask = mask[:,:,0]

        mask = (mask>0).astype(np.uint8)

        return mask

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):

        name = self.files[idx]

        img = self.read_img(os.path.join(self.img_dir,name))
        mask = self.read_mask(os.path.join(self.mask_dir,name))

        h,w = img.shape[:2]

        img_resize = self.transform.apply_image(img)

        img_tensor = torch.tensor(img_resize).permute(2,0,1).float()

        img_tensor = (img_tensor-self.pixel_mean)/self.pixel_std

        hh,ww = img_tensor.shape[-2:]

        img_tensor = F.pad(img_tensor,(0,1024-ww,0,1024-hh))

        mask = torch.tensor(mask).float()

        return img_tensor,mask,torch.tensor([h,w])


# ==============================
# Sampling
# ==============================

def sample_internal(mask):

    coords = torch.stack(torch.where(mask>0),dim=1)

    if len(coords)==0:
        return None

    y,x = coords[random.randint(0,len(coords)-1)]

    y += random.randint(-POINT_JITTER,POINT_JITTER)
    x += random.randint(-POINT_JITTER,POINT_JITTER)

    y = torch.clamp(y,0,mask.shape[0]-1)
    x = torch.clamp(x,0,mask.shape[1]-1)

    return [x.item(),y.item()]


# ==============================
# Loss
# ==============================

def dice_loss(pred,target):

    pred = torch.sigmoid(pred)

    inter = (pred*target).sum()

    union = pred.sum()+target.sum()+1e-6

    return 1-2*inter/union


# ==============================
# Load SAM
# ==============================

sam = sam_model_registry["vit_b"](checkpoint=MODEL_PATH)

sam.to(device)

sam.train()

for p in sam.image_encoder.parameters():
    p.requires_grad=False

for p in sam.prompt_encoder.parameters():
    p.requires_grad=False

optimizer = torch.optim.Adam(
    sam.mask_decoder.parameters(),
    lr=LR
)


# ==============================
# Dataset
# ==============================

dataset = SEMDataset(DATA_IMG,DATA_MASK)

loader = DataLoader(dataset,batch_size=1,shuffle=False)


# ==============================
# Cache embeddings
# ==============================

print("caching embeddings...")

embeddings=[]
masks=[]

sam.image_encoder.eval()

with torch.no_grad():

    for img,mask,size in tqdm(loader):

        emb = sam.image_encoder(img.to(device))

        embeddings.append(emb.cpu())
        masks.append(mask)

print("cache done")


# ==============================
# Training
# ==============================

indices = list(range(len(embeddings)))

for epoch in range(EPOCHS):

    random.shuffle(indices)

    total_loss=0

    for idx in tqdm(indices):

        emb = embeddings[idx].to(device)

        mask = masks[idx][0]

        h,w = mask.shape

        points=[]
        labels=[]

        for _ in range(POS_INTERNAL):

            p = sample_internal(mask)

            if p:
                points.append(p)
                labels.append(1)

        points = torch.tensor(points).float()
        labels = torch.tensor(labels).float()

        scale_x = 1024/w
        scale_y = 1024/h

        points[:,0]*=scale_x
        points[:,1]*=scale_y

        points = points.unsqueeze(0).to(device)
        labels = labels.unsqueeze(0).to(device)

        with torch.no_grad():

            sparse,dense = sam.prompt_encoder(
                points=(points,labels),
                boxes=None,
                masks=None
            )

        low_res,_ = sam.mask_decoder(
            image_embeddings=emb,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False
        )

        up = F.interpolate(
            low_res,
            size=(h,w),
            mode="bilinear",
            align_corners=False
        )

        gt = mask.unsqueeze(0).unsqueeze(0).to(device)

        bce = torch.nn.BCEWithLogitsLoss()

        loss = bce(up,gt) + dice_loss(up,gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()

    print("epoch",epoch,"loss",total_loss/len(indices))


# ==============================
# Save model
# ==============================

save_path = os.path.join(CHECKPOINT_DIR,"sam_sem_finetune_v1.pth")

torch.save(sam.state_dict(),save_path)

print("training finished")