import os
import cv2
import tqdm
import argparse
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pylab as plt

# parser = argparse.ArgumentParser(
#     description='input directory, video filetypes for reading data')
# parser.add_argument('--input_dir', default='', type=str,
#                     help='input directory')

# parser.set_defaults(os.getcwd())
# args = parser.parse_args()


folder_path='video_raw/2801_onp/'
csv_filename = "signal_list_samsung"
video_list = []
ppg_signals_list=[]


def video2Frame(file):
    count = 0
    cap = cv2.VideoCapture(file)    
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    n = 32       # Images Pixel Resize 32X32 
    dim = (n,n)
    ppg_signal=[]

    pbar = tqdm(total=length)            

    while(count!=length):

        pbar.update(1)      

        frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_red = img.copy()
        img_red[:, :, 1:3] = 0     #Getting only Red channel values

        resized = cv2.resize(img_red, dim)
        (h, w) = resized.shape[:2]
        center = (w / 2, h / 2)
        angle90 = -90
        scale = 1.0

        M = cv2.getRotationMatrix2D(center, angle90, scale)
        rotated90 = cv2.warpAffine(resized, M, (h, w))
        r = rotated90.flatten()
        avg_r= np.average(r)                
        ppg_signal.append(avg_r)              #
        count+=1
        #saveRGBImage('r_channel/r_d'+str(count)+'.jpg', rotated90)

    pbar.close()
    return ppg_signal


def preprocess():

    for f in glob(folder_path+'*.mp4'):
        video_list.append(f)
    print("Total Number of video files for PPG conversion: ",len(video_list))
    print("\n")
    print("Started converting video to frame and generating all PPG signals.......")

    for signal in video_list:
        temp = video2Frame(signal)
        ppg_signals_list.append(temp)

    print("Generating all PPG signals and saving it to"+csv_filename+".csv file.......")

    df = pd.DataFrame(ppg_signals_list)
    df.to_csv(folder_path+csv_filename+'.csv', index=False)


# # Visualize the signal from saved PPG Signal values

# signal = pd.read_csv(folder_path+csv_filename+'.csv', delimiter = ",")
# signalT=signal.T

# plt.plot(signalT[7])
# plt.xlabel('RAW PPG GRAP H')
# plt.ylabel('intensity')
# plt.show()


if __name__ == '__main__':
    preprocess()