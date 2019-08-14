import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image


def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here


    FLAGS = {
        'model': "./model_data/yolo.h5",
        'anchors': YOLO.get_defaults("anchors_paths"),
        'classes': YOLO.get_defaults("classes_path"),
        'gpu_num': YOLO.get_defaults("gpu_num"),
        'image': False,
        'input': "./data/visor_video.wmv",
        'output': "./video_output.wmv"
    }

    if FLAGS['image']:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS['input'] + "," + FLAGS['output'])
        detect_img(YOLO(**FLAGS))
    elif "input" in FLAGS:
        detect_video(YOLO(**FLAGS), FLAGS['input'], FLAGS['output'])
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
