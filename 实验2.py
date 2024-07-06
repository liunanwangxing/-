import time
import heapq
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

def center_point(img,img_color):
    img2 = img * 255
    _, binary_image = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)

    #三通道图像转化为单通道图像
    binary_image = binary_image.astype(np.uint8)

    # 寻找轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # 计算每个轮廓的面积，并保留最大的两个轮廓
    contour_areas = [(cv2.contourArea(contour), contour) for contour in contours]
    contour_areas.sort(key=lambda x: x[0], reverse=True)
    largest_contours = [contour_areas[i][1] for i in range(min(2, len(contour_areas)))]
    points = []

    # 遍历最大的两个轮廓，计算并绘制质心
    for contour in largest_contours:
        # 计算轮廓的矩
        M = cv2.moments(contour)

        # 计算质心坐标
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        points.append(cX)
        points.append(cY)

        # 在输出图像上绘制质心
        cv2.circle(img_color, (cX, cY), 1, (0, 0, 255), -1)
        cv2.putText(img_color, f"({cX}, {cY})", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


    for contour in largest_contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        print(f"Center: ({cX}, {cY})")
    return points
def close(x,y,w,h):
    global img_weighting,height,width
    for i in range(w):
        for j in range(h):
            if img_weighting[y + j, x + i] == 1:
                continue
            s1 = s2 = s3 = s4 = s5 = s6 = s7 = s8 = \
                0
            for k in range(50):
                if y+j+k>=height-1 or y+j-k<1 or x+i+k>=width-1 or x+i-k<1:
                    continue
                if img_weighting[y + j + k, x + i] == 1:
                    s1 = 1
                if img_weighting[y + j - k, x + i] == 1:
                    s2 = 1
                if img_weighting[y + j, x + i + k] == 1:
                    s3 = 1
                if img_weighting[y + j, x + i - k] == 1:
                    s4 = 1
                if img_weighting[y + j + k, x + i + k] == 1:
                    s5 = 1
                if img_weighting[y + j - k, x + i + k] == 1:
                    s6 = 1
                if img_weighting[y + j + k, x + i - k] == 1:
                    s7 = 1
                if img_weighting[y + j - k, x + i - k] == 1:
                    s8 = 1
            #
            if s1+s2+s3+s4+s5+s6+s7+s8 >= 6:
                img_weighting[y + j, x + i] = 1
#计算最大联通区域所占百分比
def calculate_largest_components_percentage(img):
    # 二值化处理
    img = img*255
    _, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # 三通道图像转化为单通道图像
    binary_image = binary_image.astype(np.uint8)
    # 计算连通域
    num_labels, labels_im = cv2.connectedComponents(binary_image)

    # 计算每个连通域的面积
    component_areas = []
    for label in range(1, num_labels):  # 忽略背景（标签0）
        area = np.sum(labels_im == label)
        component_areas.append((area, label))

    # 按面积排序，找到前两个最大的连通域
    component_areas.sort(reverse=True, key=lambda x: x[0])
    largest_areas = component_areas[:2]

    # 计算前两个最大连通域所占的百分比
    total_area = sum(area for area, _ in component_areas)
    largest_percentage = sum(area for area, _ in largest_areas) / total_area * 100
    return largest_percentage

if __name__ == '__main__':
    start_time = time.perf_counter()
    image = '8.jpeg'
    img = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
    img_color = cv2.imread(image,cv2.IMREAD_COLOR)
    img_k = img
    face_eg = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_eg.detectMultiScale(img)
    # x = y = w = h = 0
    # x1 = y1 = w1 = h1 = 0

    #x,y是左上角的点
    for(x,y,w,h) in faces:
        x1 = int(x+0.2*w)
        x2 = int(x + 0.8*w)
        y1 = int(y+0.23*h)
        y2 = int(y+0.50*h)
        # img_k = cv2.rectangle(img_k,(x1,y1),(x2,y2),(0,255,0),2)
        # print(x,y,w,h)

    #二值化了一下图像
    _, binary_image = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    #边缘检测
    edges = cv2.Canny(img, threshold1=160, threshold2=250)
    edges2 = cv2.Canny(binary_image, threshold1=230, threshold2=250)

    im_shape = img.shape
    height = im_shape[0]
    width = im_shape[1]

    #生成第一个加权矩阵
    img_weighting = np.zeros([height, width])
    weighting(edges,edges2,x1,y1,int(0.6*w),int(0.27*h))


    # 定义膨胀操作的结构元素（核）
    kernel = np.ones((2, 2), np.uint8)

    # 应用膨胀操作
    # cv2.imshow('img3', edges)
    # cv2.imshow('img5', edges2)
    # cv2.imshow('img6', binary_image)
    # cv2.imshow('img2',dilated_image)
    # cv2.imshow('img',img_weighting)
    #
    # center_point(dilated_image,img_color)
    # cv2.imshow('img4', img_color)
    # cv2.imwrite('cc.jpg', img_weighting*255)
    cv2.imshow('img1',img_weighting)
    close(x1,y1,int(0.6*w),int(0.27*h))

    cv2.imshow('img2',img_weighting)

    n = 0
    while True:
        img_weighting = cv2.dilate(img_weighting, kernel, iterations=1)
        print(calculate_largest_components_percentage(img_weighting))
        n=n+1
        if calculate_largest_components_percentage(img_weighting) >85 or n>=10:
            break


    cv2.imshow('img3', img_weighting)
    center_point(img_weighting, img_color)
    cv2.imshow('img4', img_color)
    #窗口尺寸定义
    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('img', 650, 750)


    end_time = time.perf_counter()

    # 计算程序运行时间（以秒为单位）
    run_time = end_time - start_time

    print(f"程序运行时间为：{run_time}秒")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
