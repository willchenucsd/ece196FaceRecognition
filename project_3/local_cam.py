"""
ECE196 Face Recognition Project
Author: W Chen

What this script should do:
0. assume the local host is already connected to ec2 instance.
1. start running the camera.
2. detect a face, display it, and get confirmation from user.
3. send it for classification and fetch result.
4. show result on face display.
"""

import cv2, os, time, subprocess
from picamera import PiCamera
from picamera.array import PiRGBArray

# TODO: declare useful paths here if you plan to use them
CASCADE_PATH = ""
EC2_IP = ""  # ec2
KEY_PAIR_PATH = ""  # local
NEW_FACE_PATH = ""
NEW_FACE_NAME = "new_face.jpg"
IMG_DEST_DIR = "" # ec2
FONT = cv2.FONT_HERSHEY_SIMPLEX
RESULT_SRC_DIR = ""  # ec2
RESULT_DEST_DIR = ""  # local
NEW_RESULT_PATH = ""  # local


def send_file(file_path):
    """
    send file to ec2
    :param file_path: path of local file
    """
    send_command = ["scp", "-i", KEY_PAIR_PATH, file_path, EC2_IP + ":" + IMG_DEST_DIR]
    subprocess.call(send_command)
    print 'sending file to server'
    return


def fetch_file():
    """
    use a loop to check if new result is available and if yes fetch it
    :return: local path for result
    """
    fetch_command = ["scp", "-i", KEY_PAIR_PATH, EC2_IP + ":" + RESULT_SRC_DIR, RESULT_DEST_DIR]
    while True:
        try:
            subprocess.check_call(fetch_command)
            return NEW_RESULT_PATH
        except subprocess.CalledProcessError:
            continue


def read_result(result_path):
    """
    result result and extract name and confidence
    :param result_path: path of the result file
    :return: name and confidence
    """
    result_file = open(result_path, 'r')
    result = result_file.read()
    details = result.split(',')
    name = details[0]
    conf = details[1]
    print name, conf
    return name, conf


def main():
    # 1. start running the camera.
    # TODO: initialize face detector

    # TODO: initialize camera and update parameters
    camera = PiCamera()
    width = 640
    height = 480
    camera.rotation = 180
    camera.resolution = (width, height)
    rawCapture = PiRGBArray(camera, size=(width, height))

    # warm up
    print 'Let me get ready ... 2 seconds ...'
    time.sleep(2)
    print 'Starting ...'

    # 2. detect a face, display it, and get confirmation from user.
    for f in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
        frame = f.array
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # TODO: use face detector to get faces

        for (x, y, w, h) in faces:
            print '=================================='
            print 'Face detected!'
            cv2.imshow('Face Image for Classification', frame)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            cv2.waitKey(1)
            cv2.waitKey(1)
            cv2.waitKey(1)
            answer = input('Confirm image (1-yes / 0-no): ')
            print '=================================='

            if answer == 1:
                # TODO: save face to local

                # TODO: send it for classification and fetch result.

                os.remove(face_path)
                print 'Let\'s see who you are...'
                # look for new result
                for i in range(3):
                    time.sleep(1)
                    print 'Still thinking...'

                # TODO: get new result path and get name and confidence

                print 'New result found!'

                # TODO: display on face image

                cv2.putText(frame, result_to_display, (10, 30), FONT, 1, (0, 255, 0), 2)
                cv2.imshow('Face Image for Classification', frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                cv2.waitKey(1)
                cv2.waitKey(1)
                cv2.waitKey(1)
                # remove result
                os.remove(NEW_RESULT_PATH)
                break

        rawCapture.truncate(0)
        print 'Waiting for image...'
        time.sleep(1)
    return


if __name__ == '__main__':
    main()
