# _*_ coding utf-8 _*_
# Designer: はなちゃん
# Time: 2022/5/8 11:39
# Name: sampler.py

import cv2 as cv
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from time import sleep
import os


def cut_with_xywh(frame, pic):
    return frame[pic[1]:pic[1] + pic[3], pic[0]:pic[0] + pic[2]]


def frame_with_xywh(pic):
    return (pic[0], pic[1]), (pic[0] + pic[2], pic[1] + pic[3])


def check_pictures(path):
    current = 1
    isExist = True
    # 先判断当前文件夹中已经采样了多少张图片，从最后一张之后的序号开始
    while isExist:
        if Path(path + f"/{current}.jpg").exists():
            current += 1
        else:
            isExist = False
    return current


def eye_capture(frame):
    height = frame.shape[0]
    width = frame.shape[1]

    face_classifier = cv.CascadeClassifier(r".\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml")
    faces = face_classifier.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=3)

    if len(faces) > 0:
        for face in faces:
            x, y, w, h = face
            ratio = w * h / (width * height)
            distance = 6 / ratio
            center = (x + w / 2, y + h / 2)
            if distance < 120:
                if width * 0.25 <= center[0] <= width * 0.75:
                    cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0))
                    eye_frame = cut_with_xywh(frame, face)
                    eye_classifier = cv.CascadeClassifier(
                        r".\venv\Lib\site-packages\cv2\data\haarcascade_eye_tree_eyeglasses.xml")
                    eyes = eye_classifier.detectMultiScale(eye_frame, scaleFactor=1.2, minNeighbors=3)
                    if len(eyes) == 2:
                        flag = True
                        if eyes[0][0] < eyes[1][0]:
                            left_eye = np.array(eyes[0])
                            right_eye = np.array(eyes[1])
                        else:
                            left_eye = np.array(eyes[1])
                            right_eye = np.array(eyes[0])
                        frame_left_eye = cut_with_xywh(eye_frame, left_eye)
                        frame_left_eye = cv.resize(frame_left_eye, dsize=(120, 120))
                        frame_left_eye_gray = cv.cvtColor(frame_left_eye, cv.COLOR_BGR2GRAY)
                        show_left = cv.resize(frame_left_eye_gray, dsize=(300, 300))
                        plt.imshow(show_left, cmap="gray")
                        plt.show()
                        frame_right_eye = cut_with_xywh(eye_frame, right_eye)
                        frame_right_eye = cv.resize(frame_right_eye, dsize=(120, 120))
                        frame_right_eye_gray = cv.cvtColor(frame_right_eye, cv.COLOR_BGR2GRAY)
                        show_right = cv.resize(frame_right_eye_gray, dsize=(300, 300))
                        plt.imshow(show_right, cmap="gray")
                        plt.show()
                        return frame_left_eye_gray, frame_right_eye_gray
                    else:
                        return None
                else:
                    print("头部位置有点偏向两侧了，请及时调整，多谢合作！")
                    return None
            else:
                print("头部与屏幕的距离稍微近一点哦！")
                return None
    else:
        return None


if __name__ == '__main__':
    if not Path("./eye_samples/left/left_eye").exists():
        os.mkdir("./eye_samples")
        os.makedirs("./eye_samples/left/left_eye")
        os.makedirs("./eye_samples/left/right_eye")
        os.makedirs("./eye_samples/right/left_eye")
        os.makedirs("./eye_samples/right/right_eye")
        os.makedirs("./eye_samples/middle/left_eye")
        os.makedirs("./eye_samples/middle/right_eye")

    capture = cv.VideoCapture(0)
    print("来自lyx的采样前温馨提示：采样系统目前可能比较不智能，因此采样过程中遇到任何突发的bug，请随时在输入框中输入stop进行退出"
          "，并及时与开发者（也就是我）联系。"
          "预祝您使用愉快！")
    sleep(5)

    left_current = check_pictures("./eye_samples/left/left_eye")
    right_current = check_pictures("./eye_samples/right/left_eye")
    middle_current = check_pictures("./eye_samples/middle/left_eye")

    print("你有3秒来调整眼睛的注视位置")
    sleep(3)

    while True:
        ret, frame = capture.read()
        frame = cv.flip(frame, 1)

        retval = eye_capture(frame)
        if retval is not None:
            frame_left_eye_gray, frame_right_eye_gray = retval
            key = input("请选择是否保存（按s保存）: ")
            if key == "stop":
                print("正在终止采样程序…………")
                sleep(2)
                break
            if key == 's':
                print("请选择当前采样的注视点位置：\na~屏幕左侧\ns~屏幕中部\nd~屏幕右侧")
                position = input()
                if position == "stop":
                    break
                # 注视左侧的样本：
                if position == 'a':
                    cv.imwrite(f"./eye_samples/left/left_eye/{left_current}.jpg",
                               frame_left_eye_gray)
                    cv.imwrite(f"./eye_samples/left/right_eye/{left_current}.jpg",
                               frame_right_eye_gray)
                    left_current += 1
                # 注视中间的样本：
                elif position == 's':
                    cv.imwrite(f"./eye_samples/middle/left_eye/{middle_current}.jpg",
                               frame_left_eye_gray)
                    cv.imwrite(f"./eye_samples/middle/right_eye/{middle_current}.jpg",
                               frame_right_eye_gray)
                    middle_current += 1
                # 注视右侧的样本：
                elif position == 'd':
                    cv.imwrite(f"./eye_samples/right/left_eye/{right_current}.jpg",
                               frame_left_eye_gray)
                    cv.imwrite(f"./eye_samples/right/right_eye/{right_current}.jpg",
                               frame_right_eye_gray)
                    right_current += 1
                else:
                    print("您输错字母了哟，本次采样作废。。。")
        print("采样继续")
        print("你有3秒来调整眼睛的注视位置")
        sleep(3)
