import os
from os.path import split
from os.path import splitext
import csv
from collections import defaultdict
import numpy as np
from scipy.stats.stats import pearsonr
from moviepy.editor import VideoFileClip
import time
import datetime
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import math

def extract(openface_path, file_path):
    # run extract command using opencv to extract features
    # write result to file in file_path

    video_name = splitext(split(file_path)[1])[0]
    person_name = split(split(file_path)[0])[1]

    parameters = '-vis-aus -out_dir data/' + person_name
    command = '{} -f {} {}'.format(openface_path, file_path, parameters)
    print("Start extracting features using OpenFace of %s"%(file_path))

    print("\033[95mCommand: \033[00m" , command)
    #output = os.popen(command).read()
    os.system(command)
    #print(output)

def get_feature(openface_path, file_path, csvpath = '', overwrite = False):
    # read feature csv file and convert to 190 pearsonr distances of 20 features
    # if csv not exists, run extract
    feature_20 = ["AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
                  "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
                  "AU20_r", "AU23_r", "AU25_r", "AU26_r", "pitch",
                  "roll", 'mouthH', "mouthD"]
    video_name = splitext(split(file_path)[1])[0]
    person_name = split(split(file_path)[0])[1]

    if csvpath == '':
        csvpath = 'data/{}/{}.csv'.format(person_name, video_name)
    if not os.path.exists(csvpath) or overwrite:
        extract(openface_path, file_path)
    #else:
       # print("Exists %s csv file, skip!"%video_name, end = '  ')

    with open(csvpath, 'r') as f:
      reader = csv.DictReader(f)
      i = 0
      my_dict = defaultdict(list)
      for row in reader:
          if row[' success'] == " 1":
              for k,v in row.items():
                  my_dict[k.strip(' ')].append(float(v))
    feature_dict = {}
    for i in feature_20:
        if "AU" in i:
            feature_dict[i] = my_dict[i]
    feature_dict['pitch'] = my_dict['pose_Rx']
    feature_dict['roll'] = my_dict['pose_Rz']
    feature_dict['mouthH'] = (np.array(my_dict['X_54']) - np.array(my_dict['X_48'])).tolist()
    feature_dict['mouthD'] = (np.array(my_dict['Y_62']) - np.array(my_dict['Y_66'])).tolist()
    feature_190 = []
    for i in range(len(feature_20)-1):
        for j in range(i+1, len(feature_20)):
            if sum(feature_dict[feature_20[i]]) == 0 or sum(feature_dict[feature_20[j]]) == 0:
                pp = 0
            else:
                pp = pearsonr(feature_dict[feature_20[i]], feature_dict[feature_20[j]])[0]
            #print(pp)
            feature_190.append(pp)
            # np.corrcoef
            # pearsonr
            # calcPearson
    return feature_190, feature_dict

def test_cut(video_path, target_path = 'clips/' , t1 = 0, t2 = 0):
    #cut video to clips 5 frame per cut from t1 to t2
    #the video duration is 10 second
    video_name, _ = splitext(split(video_path)[1])
    person_name = split(split(video_path)[0])[1]
    myclip = VideoFileClip(video_path)
    target_path = target_path + person_name + '/' + video_name + '/'
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    print('fps:', myclip.fps, 'frames per second')
    print('duration:', myclip.duration, 'second')
    print('clip duration:', t1, 'to', t2, ' sec')
    if t2 >= myclip.duration:
        t2 = myclip.duration
    if t1 <= 0 or t1 >= myclip.duration:
        raise ValueError("Please give a correct t1!")
    starting_time =  time.time()
    i = 0
    j = 1
    count = 1
    for frame in myclip.iter_frames( with_times = True):
        time_of_frame = frame[0]
        if time_of_frame >= t1 and time_of_frame + 10 <= t2:
            if i % 5 == 0:
                filename = video_name + '_' + '%.2f' % time_of_frame + '.mp4'
                filepath = target_path + filename
                count += 1
                if not os.path.exists(filepath):
                    newclip = myclip.subclip(time_of_frame, time_of_frame + 10)
                    newclip.write_videofile(filepath, fps = myclip.fps, preset= "medium", logger=None,
                        ffmpeg_params=['-crf', str(20)])
                    print(filename, end = ' ')
            i += 1
        if time_of_frame> j *10:
            #time_taken = str(datetime.timedelta(0, time.time() - starting_time))
            #print("Written clips {}s-{}s to file: {} in {}".format('%.2f' %time_of_frame,
            #    '%.2f' %(time_of_frame + 10), filename, time_taken))
            print("Processed %s sec of videos                       "%(time_of_frame), end="\r", flush=True)
            j += 1

        elif time_of_frame + 10 > t2:
            break

    print("\nProduced %s clips in directory %s" %(count, target_path))

def get_tSNE_of_clips(full_clip_path = '', number_of_clips = -1, video_path =  ''):
    # Get tSNE of clips.
    # Still testing because some pearsonr is nan

    clip_path = 'clips/'
    if full_clip_path == '':
        video_name, _ = splitext(split(video_path)[1])
        person_name = split(split(video_path)[0])[1]
        full_clip_path = clip_path + person_name + '/' + video_name + '/'
    clip_list = os.listdir(full_clip_path)

    if number_of_clips < 0:
        number_of_clips = len(clip_list)
    count = 0
    for file in clip_list:
        if count < number_of_clips:
            file_path = full_clip_path + file
            feature_190, _ = get_feature(openface_path, file_path)
            feature_190 = np.array(feature_190).reshape(-1, 1)
            if np.isnan(feature_190).any():
                feature_190, _ = get_feature(openface_path, file_path, overwrite = True)
                feature_190 = np.array(feature_190).reshape(-1, 1)
        else:
            break

        if count == 0:
            features = feature_190
        elif count < number_of_clips:
            features = np.concatenate((features, feature_190), axis = 1)
        count += 1

        #print(count, features.shape)
    ts = TSNE(n_components=2)
    #print(features)
    y = ts.fit_transform(features)

    #print(y)
    return y

def show_tSNE(y, color):
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(2, 1, 2)
    plt.scatter(y[:, 0], y[:, 1], c='red', cmap=plt.cm.Spectral)
    ax1.set_title('t-SNE Curve', fontsize=14)
    plt.show()

openface_path = 'C:/Users/Jason/OpenFace/FaceLandmarkVidMulti.exe'
video_path = 'src/videos/Hillary_Clinton/d.webm'
#'src/videos/Hillary_Clinton/c.mp4'
#'src/videos/Kate_McKinnon/b.mp4'



target_path = 'clips/'
#test_cut(video_path, target_path = 'clips/' , 95, 125)
y1 = get_tSNE_of_clips("",50, video_path = video_path)
y2 = get_tSNE_of_clips("",50, video_path = 'src/videos/Donald_Trump/002.mp4')
fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(2, 1, 2)
plt.scatter(y1[:, 0], y1[:, 1], c='red', cmap=plt.cm.Spectral)
plt.scatter(y2[:, 0], y2[:, 1], c='blue', cmap=plt.cm.Spectral)
ax1.set_title('t-SNE Curve', fontsize=14)
plt.show()
