import torch, torch.nn as nn
from torch.utils.data import DataLoader
from model import get_model
from datasets.uadfv import UADFV

ds = UADFV('data/UADFV', split='train')
dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)

net = get_model().cuda()
crit = nn.CrossEntropyLoss()
opt = torch.optim.AdamW(net.parameters(), lr=3e-4)

for epoch in range(5):
    for x,y in dl:
        x,y = x.cuda(), y.cuda()
        opt.zero_grad()
        loss = crit(net(x), y)
        loss.backward()
        opt.step()
    print(f'epoch {epoch} loss={loss.item():.4f}')