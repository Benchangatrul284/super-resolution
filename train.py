import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from dataset import DIV2K, FLR2K, Butterfly
import matplotlib.pyplot as plt
from model import IMDN ,IMDN_RTC, IMDN_SA
import math
from demo import demo_UHD_fast,psnr_tensor, TestSet
from best import IMDN_DW
from own_model import IMDN_MH
# plot loss curve
def plot_loss_curve():
    plt.cla()
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x.get('val_loss') for x in history]
    plt.plot(train_losses, '-bx',label='train')
    plt.plot(val_losses, '-rx',label='val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs')
    plt.legend()
    plt.savefig('loss_curve.png')
# plot psnr curve
def plot_psnr_curve():
    plt.cla()
    psnrs = [x.get('test_psnrs') for x in history]
    plt.plot(psnrs, '-bx')
    plt.xlabel('epoch')
    plt.ylabel('psnr')
    plt.title('PSNR vs. No. of epochs');
    plt.savefig('psnr_curve.png')
# plot lr curve
def plot_lr_curve():
    plt.cla()
    lrs = [x.get('lrs') for x in history]
    plt.plot(lrs, '-bx')
    plt.xlabel('epoch')
    plt.ylabel('lr')
    plt.title('lr vs. No. of epochs');
    plt.savefig('lr_curve.png')

parser = argparse.ArgumentParser()
parser.add_argument('--nEpochs', type=int, default=3000, help='number of epochs to train for')
parser.add_argument('--resume', type=str, default='', help='path to checkpoint')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--export', type=str, default='model.pth')
args = parser.parse_args()

# adjust learning rate
def adjust_learning_rate(epoch, T_max=1000, eta_min=2e-4, lr_init=args.lr):
    lr = eta_min + (lr_init - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2
    if epoch >= T_max:
        lr = eta_min
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train():
    global history
    train_loss = 0
    best_psnr = 0
    start_epoch = 1
    #loading pretrained models
    if args.resume:
        if os.path.isfile(args.resume):
            print("===> loading models '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model'])
            start_epoch = checkpoint['epoch']
            best_psnr = checkpoint['best_psnr']
            history = checkpoint['history']
            print("checkpoint loaded: epoch = {}, PSNR = {}".format(start_epoch, best_psnr))
        else:
            print("===> no models found at '{}'".format(args.resume))

    model.train()
    for epoch in range(start_epoch,epochs + 1):
        adjust_learning_rate(epoch)
        result = {'train_loss': [], 'val_loss': [], 'lrs': [], 'test_psnrs': []}
        print('Epoch: {}'.format(epoch))
        print('learning rate: {:.6f}'.format(optimizer.param_groups[0]['lr']))
        for (img_hr, img_lr) in tqdm(train_dl):
            img_hr = img_hr.to(device)
            img_lr = img_lr.to(device)
            optimizer.zero_grad()
            output = model(img_lr)
            loss = criterion(output, img_hr)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_dl)
        
        model.eval()
        with torch.no_grad():
            # compute validation loss 
            valid_loss = 0
            for (img_hr, img_lr) in tqdm(valid_dl):
                img_hr = img_hr.to(device)
                img_lr = img_lr.to(device)
                output = model(img_lr)
                loss = criterion(output, img_hr)
                valid_loss += loss.item()
            valid_loss = valid_loss / len(valid_dl)

            # compute test PSNR
            total_psnr=0
            for data in test_dl:
                img_lq = data[1].to(device) #whole LR image, RGB
                img_gt = data[2].to(device) #whole HR image, RGB
                _, _, h_old, w_old = img_lq.size()
                h_pad = (2 ** math.ceil(math.log2(h_old))) - h_old
                w_pad = (2 ** math.ceil(math.log2(w_old))) - w_old
                img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
                img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
                output = demo_UHD_fast(img_lq, model) # whole SR image, RGB
                preds = (output[:, :, :h_old*3, :w_old*3].clamp(0, 1) * 255).round() # 1 -> 255
                # [1,3,276,276]
                img_gt = (img_gt[:, :, :h_old*3, :w_old*3] * 255.).round() # 0~1 -> 0~255
                total_psnr+= psnr_tensor(preds, img_gt).item()
        avg_psnr = total_psnr/len(test_dl)
        result['train_loss'].append(train_loss)
        result['val_loss'].append(valid_loss)
        result['test_psnrs'].append(avg_psnr)
        result['lrs'].append(optimizer.param_groups[0]['lr'])
        print('Train Loss: {:.4f}'.format(train_loss))
        print('Val Loss: {:.4f}'.format(valid_loss))
        print('Test PSNR: {:.4f}'.format(avg_psnr))
        history.append(result)

        if avg_psnr > best_psnr:
            model_folder = "checkpoint"
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
            best_psnr = avg_psnr # update best PSNR
            model_out_path = os.path.join(model_folder, args.export)
            state = {"epoch": epoch,
                    "model": model.state_dict(),
                    "best_psnr": best_psnr,
                    "history": history}
            torch.save(state, model_out_path)
            print("===> Checkpoint saved to {}".format(model_out_path))

        plot_loss_curve()
        plot_psnr_curve()
        plot_lr_curve()

if __name__ == '__main__':
    # training settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = IMDN_MH().to(device)
    criterion = nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    epochs = args.nEpochs
    div2k_train = DIV2K(root='DIV2K_decoded', repeat=20, patch_size=255, mode='train')
    flickr2k = FLR2K(root='Flickr2K_decoded', repeat=1, patch_size=255)
    butterfly = Butterfly(root='Butterfly_decoded', repeat=50, patch_size=96)
    train_ds= torch.utils.data.ConcatDataset([div2k_train, flickr2k])

    train_dl = DataLoader(dataset = div2k_train, num_workers=6, batch_size=16, shuffle=True, pin_memory=True, drop_last=True)
    valid_ds = DIV2K(root='DIV2K_decoded', repeat=10, patch_size=255, mode='val')
    valid_dl = DataLoader(dataset=valid_ds, num_workers=6, batch_size=10, shuffle=False, pin_memory=True, drop_last=True)
    test_ds = TestSet(lq_paths=['Set5/LRbicx3'], gt_paths=['Set5/original'])
    test_dl = DataLoader(test_ds, batch_size=1)
    history = []
    train()