#!/usr/bin/env python3

import os
import cv2
import sys

root_dir = "~/Programming/Python/tinder-bot/faces"

# print(root_dir)

faces_dir = './faces'

cv2.namedWindow('image', cv2.WINDOW_NORMAL)

dataset = []

for root, dirs, files in os.walk(faces_dir):
    for dirname in dirs:
        for root, dirs, files in os.walk(os.path.join(faces_dir, dirname)):
            for image in files:
                image_fullpath = os.path.join(faces_dir, dirname, image)
                print(image_fullpath)
                img = cv2.imread(image_fullpath, cv2.IMREAD_COLOR)
                cv2.imshow('image', img)
                k = cv2.waitKey(0)
                print(k)
                if k == 121: # 'y'
                    like = True
                elif k == 110: # 'f'
                    like = False
                elif k == 27:
                    cv2.destroyAllWindows()
                    print(dataset)
                    sys.exit() 
                dataset.append({'path': image_fullpath, 'preference': like})

print(dataset)


# cv2.waitKey(0)
# img = cv2.imread('./faces/Aaron_Eckhart/Aaron_Eckhart_0001.jpg', cv2.IMREAD_COLOR)
# cv2.imshow('image', img)
# cv2.waitKey(0)
# img = cv2.imread('./faces/Audrey_Lacroix/Audrey_Lacroix_0001.jpg', cv2.IMREAD_COLOR)
# cv2.imshow('image', img)
# cv2.waitKey(0)
cv2.destroyAllWindows()

