import time

import cv2
import numpy as np

class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y


connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), Point(0, 1), Point(-1, 1),
            Point(-1, 0),Point(2, 0),Point(2, -1),Point(2, 1),Point(2, 2),Point(2, -2),Point(1, 2),Point(1, -2),
            Point(0, 2),Point(0, -2),Point(-1, 2),Point(-1, -2),Point(-2, 0),Point(-2, 1),Point(-2, 2),Point(-2, -2),Point(-2, -1)]

def do_weighting(y,x,img):
    a = 0
    for i in range(24):
        if img[y+connects[i].y,x+connects[i].x] == 255:
            a += 1
    if a>=4:
        return 1
    else:
        return 0


#两个灰度图像算加权点的函数
def weighting(img1,img2,x,y,w,h):
    global img_weighting
    for i in range(w):
        for j in range(h):
            if img1[y+j,x+i] == 0:  #img方括号里是先Y后X
                continue
            #计算这个点的加权值，并保存到新矩阵里：
            else:
                if do_weighting(y+j,x+i,img2) == 1:
                    img_weighting[y+j,x+i] = 1





if __name__ == '__main__':
    start_time = time.perf_counter()
    img = cv2.imread('zake.jpg',cv2.IMREAD_GRAYSCALE)
    face_eg = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_eg.detectMultiScale(img)

    #x,y是左上角的点
    for(x,y,w,h) in faces:
        x1 = int(x+0.2*w)
        x2 = int(x + 0.8*w)
        y1 = int(y+0.23*h)
        y2 = int(y+0.50*h)
        # img = cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        # print(x,y,w,h)

    #二值化了一下图像
    _, binary_image = cv2.threshold(img, 35, 255, cv2.THRESH_BINARY)
    #边缘检测
    edges = cv2.Canny(img, threshold1=160, threshold2=250)
    edges2 = cv2.Canny(binary_image, threshold1=150, threshold2=250)

    im_shape = img.shape
    height = im_shape[0]
    width = im_shape[1]

    #生成第一个加权矩阵
    img_weighting = np.zeros([height, width])
    weighting(edges,edges2,0,0,int(0.9*width),int(0.9*height))

    # 定义膨胀操作的结构元素（核）
    kernel = np.ones((9, 9), np.uint8)

    # 应用膨胀操作
    dilated_image = cv2.dilate(img_weighting, kernel, iterations=1)
    # cv2.imshow('img3', edges)
    # cv2.imshow('img2',dilated_image)
    # cv2.imshow('img',img_weighting)
    cv2.imwrite("j.jpg",img_weighting*255)
    cv2.imwrite("j2.jpg",dilated_image*255)




    #窗口尺寸定义
    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('img', 650, 750)

    cv2.imshow('img',edges2)
    cv2.imshow('img2',edges)
    cv2.imshow('img3',binary_image)
    end_time = time.perf_counter()

    # 计算程序运行时间（以秒为单位）
    run_time = end_time - start_time

    print(f"程序运行时间为：{run_time}秒")

    cv2.waitKey(0)
