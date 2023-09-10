import cv2
import os
import tqdm
import torch
import numpy as np
from tqdm import tqdm
import math
import argparse
import sys

# import your model here
os.chdir(os.path.join(os.getcwd(), 'utils'))
sys.path.append('../')
from model import IMDN_MH

parser = argparse.ArgumentParser()
parser.add_argument('--org', type=str, default='org.mp4', help='path to original video')
parser.add_argument('--crop', type=str, default='crop.mp4', help='path to store crop video')
parser.add_argument('--downscale', type=str, default='downscale.mp4', help='path to store downscale video')
parser.add_argument('--SR', type=str, default='sr.mp4', help='path to store SR video')
parser.add_argument('--output',type=str,default='output.mp4',help='path to store output video')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def crop_video(video_path, crop_path, crop_size=720):
    '''
    crop the video to 720*720
    '''
    cap = cv2.VideoCapture(video_path)
    # capture frames from a video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_writer = cv2.VideoWriter(crop_path,cv2.VideoWriter_fourcc(*'mp4v'), fps, (crop_size, crop_size))

    #read the video
    for _ in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (crop_size, crop_size))
        video_writer.write(frame)
    cap.release()

def demo_UHD_fast(img, model):
    # test the image tile by tile
    # print(img.shape) # [1, 3, 2048, 1152] for ali forward data
    scale = 3
    b, c, h, w = img.size()
    tile = min(256, h, w)
    tile_overlap = 0
    stride = tile - tile_overlap
    
    h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
    w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
    E = torch.zeros(b, c, h*scale, w*scale).type_as(img)
    W = torch.zeros_like(E)
    
    in_patch = []
    # append all 256x256 patches in a batch with size = 135
    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch.append(img[..., h_idx:h_idx+tile, w_idx:w_idx+tile].squeeze(0))

    in_patch = torch.stack(in_patch, 0)
    out_patch = model(in_patch)
    
    for ii, h_idx in enumerate(h_idx_list):
        for jj, w_idx in enumerate(w_idx_list):
            idx = ii * len(w_idx_list) + jj
            out_patch_mask = torch.ones_like(out_patch[idx])

            E[..., h_idx*scale:(h_idx+tile)*scale, w_idx*scale:(w_idx+tile)*scale].add_(out_patch[idx])
            W[..., h_idx*scale:(h_idx+tile)*scale, w_idx*scale:(w_idx+tile)*scale].add_(out_patch_mask)
            
    output = E.div_(W)
    return output

def load_model(model, model_path):
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    return model

def video_downscale(crop_path,downscale_path,scale=3):
    '''
    downscale the video
    '''
    cap = cv2.VideoCapture(crop_path)
    # capture frames from a video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_writer = cv2.VideoWriter(downscale_path,cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width//scale, frame_height//scale))

    #read the video
    for _ in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (frame_width//scale, frame_height//scale))
        video_writer.write(frame)
    cap.release()

def video_SR(org_path = 'downscale.mp4',SR_path = 'SR.mp4',scale = 3):
    model = IMDN_MH().to(device)
    model = load_model(model, 'model.pth')
    model.eval()

    cap = cv2.VideoCapture(org_path)
    # capture frames from a video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scale = 3
    video_writer = cv2.VideoWriter(SR_path,cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width*scale, frame_height*scale))

    for _ in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame.astype(np.float32) / 255. # [0, 1]
        frame = np.transpose(frame if frame.shape[2] == 1 else frame[:, :, [2, 1, 0]], (2, 0, 1))   # HWC-BGR to CHW-RGB
        frame = torch.from_numpy(frame).to(device)
        frame = frame.unsqueeze(0) # [1,3,1080,1920]
        
        # frame is the input image tensor, with shape [1,3,h,w] on GPU
        with torch.no_grad():
            _, _, h_old, w_old = frame.size()
            h_pad = (2 ** math.ceil(math.log2(h_old))) - h_old
            w_pad = (2 ** math.ceil(math.log2(w_old))) - w_old
            frame = torch.cat([frame, torch.flip(frame, [2])], 2)[:, :, :h_old + h_pad, :]
            frame = torch.cat([frame, torch.flip(frame, [3])], 3)[:, :, :, :w_old + w_pad]
            output = demo_UHD_fast(frame, model) # whole SR image, RGB
            preds = (output[:, :, :h_old*3, :w_old*3].clamp(0, 1) * 255).round() # 1 -> 255
            # [1,3,276,276]
            preds = np.transpose(preds.squeeze().cpu().numpy()[[2,1,0],:,:], (1, 2, 0)) #BGR
            video_writer.write(preds.astype(np.uint8))
    cap.release()

def output_comparison(downscale_path = 'downscale.mp4',SR_path = 'sr.mp4',org_path = 'org.mp4',output_path = 'output.mp4'):
    cap_down = cv2.VideoCapture(downscale_path)
    cap_SR = cv2.VideoCapture(SR_path)
    cap_org = cv2.VideoCapture(org_path)
    # get the frame width and height of the videos
    frame_width = int(cap_SR.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_SR.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap_SR.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap_SR.get(cv2.CAP_PROP_FRAME_COUNT))
    padding_x = 50
    padding_y = 100
    # create a VideoWriter object to write the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, ((frame_width)*3+4*padding_x, frame_height+padding_y))

    for _ in tqdm(range(frame_count)):
        # read a frame from the first video
        ret1, frame1 = cap_down.read()
        if ret1:
            frame1 = cv2.resize(frame1, (frame_width, frame_height))
        # read a frame from the second video
        ret2, frame2 = cap_SR.read()
        
        ret3, frame3 = cap_org.read()
        # if either video has ended, break out of the loop
        if not ret1 or not ret2 or not ret3:
            break

        # add borders to the frames
        frame1 = cv2.copyMakeBorder(frame1, top=padding_y,left=0,bottom=0,right=padding_x,borderType=cv2.BORDER_CONSTANT)
        frame2 = cv2.copyMakeBorder(frame2, top=padding_y,left=padding_x,bottom=0,right=padding_x,borderType=cv2.BORDER_CONSTANT)
        frame3 = cv2.copyMakeBorder(frame3, top=padding_y,left=padding_x,bottom=0,right=0,borderType=cv2.BORDER_CONSTANT)
        # concatenate the frames side by side
        comparison = cv2.hconcat([frame1, frame2, frame3])


        # add text overlay to the concatenated frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        color = (0,0,255)
        thickness = 4

        # add file name of first video
        text1 = 'downscale'
        text_size1 = cv2.getTextSize(text1, font, font_scale, thickness)[0]
        text_x1 = frame_width//2 - text_size1[0]//2
        text_y1 = padding_y//2
        cv2.putText(comparison, text1, (text_x1, text_y1), font, font_scale, color, thickness)

        # add file name of second video
        text2 = 'SR'
        text_size2 = cv2.getTextSize(text2, font, font_scale, thickness)[0]
        text_x2 = (frame_width*3+4*padding_x)//2-text_size2[0]//2
        text_y2 = padding_y//2
        cv2.putText(comparison, text2, (text_x2, text_y2), font, font_scale, color, thickness)

        # add file name of third video
        text3 = 'org'
        text_size3 = cv2.getTextSize(text3, font, font_scale, thickness)[0]
        text_x3 = frame_width*3+4*padding_x-frame_width//2 - text_size3[0]//2
        text_y3 = padding_y//2
        cv2.putText(comparison, text3, (text_x3, text_y3), font, font_scale, color, thickness)

        #write the concatenated frames to the output video
        out.write(comparison)

    # release the video capture objects and the output video object
    cap_down.release()
    cap_SR.release()
    cap_org.release()
    out.release()

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    if args.crop:
        print('cropping......')
        crop_video(args.org,args.crop)
        print('crop finished')
    if args.downscale:
        print('downscaling......')
        video_downscale(args.crop,args.downscale,scale = 3)
        print('downscale finished')
    if args.SR:
        print('Super resolution......')
        video_SR(args.downscale,args.SR,scale = 3)
        print('SR finished')
    if args.output:
        print('outputing......')
        output_comparison(args.downscale,args.SR,args.crop,args.output)
        print('output finished')
   
        
