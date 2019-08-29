import sys
import you_get
import subprocess



def download(url, path, name):
    #format = '-format=mp4'
    download_path = path + name + '/'
    subprocess.call(
        'you-get -o {} {}'.format( download_path, url),
        shell=True
    )



if __name__ == '__main__':

    path = 'src/videos'
    name = 'unknown'
    list = ['https://www.youtube.com/watch?v=bx6V-e2DQW0',
            'https://www.youtube.com/watch?v=c2DgwPG7mAA',
            'https://www.youtube.com/watch?v=LNK430YOiT4',
            'https://www.youtube.com/watch?v=-nQGBZQrtT0',
            'https://www.youtube.com/watch?v=ZwQkBfBs958',
            'https://www.youtube.com/watch?v=Kbryz0mxuMY',
            'https://www.youtube.com/watch?v=O3iBb1gvehI',
            'https://www.youtube.com/watch?v=C6GnHBEBWYE']
    for url in list:
        download(url, path, name)
