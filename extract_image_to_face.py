#conding=utf8
import os
import dlib
import cv2
import argparse
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


def extract_face_from_images(dataset, compression, start, num):
    data_path = 'G:\FaceForensics++'
    images_path = os.path.join(data_path, DATASET_PATHS[dataset], compression, 'images')

    if num == 'all':
        faces_path = os.path.join(data_path, DATASET_PATHS[dataset], compression, 'faces_all')
    else:
        faces_path = os.path.join(data_path, DATASET_PATHS[dataset], compression, 'faces_'+num)

    os.makedirs(faces_path, exist_ok=True)
    face_detector = dlib.get_frontal_face_detector()

    files= os.listdir(images_path)
    pbar = tqdm(total=len(files)-start)
    for index in range(start,len(files)):
        pbar.update(1)

        filename = files[index]
        output_path = os.path.join(faces_path, filename)
        os.makedirs(output_path, exist_ok=True)
        input_path = os.path.join(images_path, filename)

        images = os.listdir(input_path)
        length = len(images)
        if num == 'all':
            step = 1
        else:
            step = int(num)d

        for i in range(0,length,step):
            imagepath = os.path.join(input_path, images[i])
            image = cv2.imread(imagepath)
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces = face_detector(gray, 1)
            if(len(faces)):
                face = get_max_face(faces)

                x, y, size = get_boundingbox(face, width, height)
                cropped_face = image[y:y+size, x:x+size]
                cv2.imwrite(os.path.join(output_path, images[i]),cropped_face)

    pbar.close()

if __name__ == "__main__":

    p = argparse.ArgumentParser()
	#p.add_argument('--data_path', type=str)
    p.add_argument('--dataset', '-d', type=str,
                   choices=list(DATASET_PATHS.keys()) + ['all'],
                   default='original')
    p.add_argument('--compression', '-c', type=str, choices=COMPRESSION + ['all'],
                   default='c23')
    p.add_argument('--start', '-s', type=int, default=0)
    p.add_argument('--num', '-n', type=str, default='10')

    args = p.parse_args()

    if args.dataset == 'all':
        if args.compression == 'all':
            for dataset in DATASET_PATHS.keys():
                args.dataset = dataset
                for compression in COMPRESSION:
                    args.compression = compression
                    extract_face_from_images(**vars(args))
        else:
            for dataset in DATASET_PATHS.keys():
                args.dataset = dataset
                extract_face_from_images(**vars(args))

    else:
        if args.compression == 'all':
            for compression in COMPRESSION:
                args.compression = compression
                extract_face_from_images(**vars(args))
        else:
            extract_face_from_images(**vars(args))
