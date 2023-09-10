import cv2
from tqdm import tqdm
def output_comparison(downscale_path = 'downscale.mp4',SR_path = 'SR.mp4',org_path = 'org.mp4',output_path = 'output.mp4'):
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

        # write the concatenated frames to the output video
        out.write(comparison)

    # release the video capture objects and the output video object
    cap_down.release()
    cap_SR.release()
    cap_org.release()
    out.release()

if __name__ == '__main__':
    output_comparison()