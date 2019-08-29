"""
Extracts images from (compressed) videos, used for the FaceForensics++ dataset

Usage: see -h or https://github.com/ondyari/FaceForensics

Author: Andreas Roessler
Date: 25.01.2019
"""
import os
from os.path import join
import argparse
import subprocess
import cv2
import dlib
from tqdm import tqdm


DATASET_PATHS = {
    'original': 'original',
    'Deepfakes': 'Deepfakes',
    'Face2Face': 'Face2Face',
    'FaceSwap': 'FaceSwap'
}
COMPRESSION = ['c0', 'c23', 'c40']


def get_boundingbox(face, width, height, scale=1.3, minsize=None):

    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    #size_bb = int(max(x2 - x1, y2 - y1))
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb

def get_max_face(faces):
    max_w = 0
    max_i = 0
    for i, face in enumerate(faces):
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
        w = min(w,h)
        if w > max_w:
            max_w = w
            max_i = i
    return faces[max_i]

def extract_frames(data_path, images_path, faces_path, method='cv2'):
    """Method to extract frames, either with ffmpeg or opencv. FFmpeg won't
    start from 0 so we would have to rename if we want to keep the filenames
    coherent."""
    #out_dir = os.path.dirname(output_path)
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(faces_path, exist_ok=True)
    if method == 'ffmpeg':
        subprocess.check_output(
            'ffmpeg -i {} {}'.format(
                data_path, join(images_path, '%04d.png')),
            shell=True, stderr=subprocess.STDOUT)
    elif method == 'cv2':
        reader = cv2.VideoCapture(data_path)
        frame_num = 0
        #frame_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        face_detector = dlib.get_frontal_face_detector()
        while reader.isOpened():
            success, image = reader.read()
            if not success:
                break
            #cv2.imwrite(join(images_path, '{:04d}.png'.format(frame_num)),image)
            
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces = face_detector(gray, 1)
            if(len(faces)):
                face = get_max_face(faces)

                x, y, size = get_boundingbox(face, width, height)
                cropped_face = image[y:y+size, x:x+size]
                cv2.imwrite(os.path.join(faces_path, '{:04d}.png'.format(frame_num)),cropped_face)

            frame_num += 1
            
        reader.release()
    else:
        raise Exception('Wrong extract frames method: {}'.format(method))


def extract_method_videos(data_path, dataset, compression, start):
    """Extracts all videos of a specified method and compression in the
    FaceForensics++ file structure"""
    videos_path = join(data_path, DATASET_PATHS[dataset], compression, 'videos')
    images_path = join(data_path, DATASET_PATHS[dataset], compression, 'images')
    faces_path = join(data_path, DATASET_PATHS[dataset], compression, 'faces_all')
    print("Extracting the", videos_path)
    videos = os.listdir(videos_path)
    end = len(videos)
    pbar = tqdm(total=end-start)
    for i in range(start, end):
        video = videos[i]
        print(video)
        image_folder = video.split('.')[0]
        extract_frames(join(videos_path, video),
                       join(images_path, image_folder), join(faces_path, image_folder))
        pbar.update(1)
    pbar.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--data_path', type=str)
    p.add_argument('--dataset', '-d', type=str,
                   choices=list(DATASET_PATHS.keys()) + ['all'],
                   default='all')
    p.add_argument('--compression', '-c', type=str, choices=COMPRESSION + ['all'],
                   default='all')
    p.add_argument('--start', '-s', type=int, default=0)
    args = p.parse_args()

    if args.dataset == 'all':
        if args.compression == 'all':
            for dataset in DATASET_PATHS.keys():
                args.dataset = dataset
                for compression in COMPRESSION:
                    args.compression = compression
                    extract_method_videos(**vars(args))
        else:
            for dataset in DATASET_PATHS.keys():
                args.dataset = dataset
                extract_method_videos(**vars(args))

    else:
        if args.compression == 'all':
            for compression in COMPRESSION:
                args.compression = compression
                extract_method_videos(**vars(args))
        else:
            extract_method_videos(**vars(args))
