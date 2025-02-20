import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from model import DeepTTG
import metrics
from datetime import datetime
import torch
from torch_geometric.loader import DataLoader
from dataset import TestbedDataset
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import GradScaler, autocast

def train(model, device, train_loader, optimizer,loss_fn):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()




        with autocast():
            output = model(data)
            loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        # break
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()



def test(model: nn.Module, test_loader, loss_function, device, show):
    model.eval()
    test_loss = 0
    outputs = []
    targets = []
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(test_loader), disable=not show, total=len(test_loader)):
            data = data.to(device)
            y = data.y
            y_hat,_,_ = model(data)
            test_loss += loss_function(y_hat.view(-1), y.view(-1)).item()
            outputs.append(y_hat.cpu().numpy().reshape(-1))
            targets.append(y.cpu().numpy().reshape(-1))

    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)

    test_loss /= len(test_loader.dataset)

    evaluation = {
        'loss': test_loss,
        'c_index': metrics.c_index(targets, outputs),
        'RMSE': metrics.RMSE(targets, outputs),
        'MAE': metrics.MAE(targets, outputs),
        'SD': metrics.SD(targets, outputs),
        'CORR': metrics.CORR(targets, outputs),
    }

    return evaluation










scaler = GradScaler()
torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True

model = DeepTTG().to("cuda")
device = torch.device("cuda")
loss_fn = nn.MSELoss(reduction='sum')
optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)
data_loaders = {phase_name:
                DataLoader(TestbedDataset(root='data', dataset=phase_name),
                           batch_size=96,
                           pin_memory=True,
                           shuffle=True)
            for phase_name in ['train', 'val', 'test2016','test2013']}

path = Path(f'result/DeepTTG_{datetime.now()}_{seed}')
writer = SummaryWriter(path)
NUM_EPOCHS = 100
save_best_epoch=80
best_val_loss = 100000000
best_epoch = -1

start = datetime.now()
print('start at ', start)

for epoch in range(1,NUM_EPOCHS+1):

    train(model, device, data_loaders['train'], optimizer, loss_fn)
    # break
    for _p in [ 'val', 'test2016','test2013']:
        performance = test(model, data_loaders[_p], loss_fn, device, False)
        for i in performance:
            writer.add_scalar(f'{_p} {i}', performance[i], global_step=epoch)
        if _p == 'val' and epoch >= save_best_epoch and performance['loss'] < best_val_loss:
            best_val_loss = performance['loss']
            best_epoch = epoch
            torch.save(model.state_dict(), path / 'best_model.pt')


model.load_state_dict(torch.load(path / 'best_model.pt'))
with open(path / 'result.txt', 'w') as f:
    f.write(f'best model found at epoch NO.{best_epoch}\n')
    for _p in ['train', 'val', 'test2016','test2013']:
        performance = test(model, data_loaders[_p], loss_fn, device, True)
        f.write(f'{_p}:\n')
        print(f'{_p}:')
        for k, v in performance.items():
            f.write(f'{k}: {v}\n')
            print(f'{k}: {v}\n')
        f.write('\n')
        print()

print('train finished')

end = datetime.now()
print('end at:', end)
print('time used:', str(end - start))
