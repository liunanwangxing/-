import time
import heapq
import cv2
import numpy as np
import math

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

connects2 = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), Point(0, 1), Point(-1, 1),
            Point(-1, 0)]
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

def find_center_point():
    global points, img_mark, center_points, center_point_totalx, center_point_totaly, center_point_total_num
    center_points = []
    for i in range(2):
        center_x = center_point_totalx[i] // center_point_total_num[i]
        center_y = center_point_totaly[i] // center_point_total_num[i]
        center_point = Point(center_x, center_y)
        center_points.append(center_point)



def grow():
    global img_expand, img_mark, points, img_mid, center_points, midpoints,height,width
    for point in points:  # 种子点就是种子点，另建一个新变量来保存已有的点
        area_num = img_mark[point.y, point.x]
        area_num = int(area_num)
        for i in range(8):
            tmpX = point.x + connects2[i].x
            tmpY = point.y + connects2[i].y


            # 超过图像范围的不要
            if tmpX < 0 or tmpY < 0 or tmpY >= height or tmpX >= width:
                continue
            # 要保证未分组且惩罚函数没有的点才能打惩罚函数
            # print(img_mark[tmpY, tmpX],img_expand[tmpY, tmpX])
            if img_mark[tmpY, tmpX] == 0 and img_expand[tmpY, tmpX] == 0:
                img_expand[tmpY, tmpX] = expand(Point(tmpX, tmpY), center_points[area_num - 1],area_num)
                # 用Point实例化对象的时候，x在前，y在后

                img_mid[tmpY, tmpX] = img_mark[point.y, point.x]
                midpoints.append(Point(tmpX, tmpY))

# 打上惩罚函数的点进行转化
def change():
    global img_expand, img_mark, img_mid, midpoints, center_point_totalx, center_point_totaly, center_point_total_num, b_list, g_list, r_list
    for midpoint in midpoints:
        img_expand[midpoint.y, midpoint.x] = img_expand[midpoint.y, midpoint.x] - 1
        if img_expand[midpoint.y, midpoint.x] <= 0:

            img_mark[midpoint.y, midpoint.x] = img_mid[midpoint.y, midpoint.x]
            img_mid[midpoint.y, midpoint.x] = 0

            # 加减点的时候，要进行中心点和的变动
            i = img_mark[midpoint.y, midpoint.x]
            i = int(i)
            center_point_totaly[i - 1] = center_point_totaly[i - 1] + midpoint.y
            center_point_totalx[i - 1] = center_point_totalx[i - 1] + midpoint.x
            center_point_total_num[i - 1] += 1

            b, g, r = img_color[midpoint.y, midpoint.x]
            b_list[i - 1] = b_list[i - 1] + b
            g_list[i - 1] = g_list[i - 1] + g
            r_list[i - 1] = r_list[i - 1] + r

            midpoints.remove(midpoint)

            points.append(midpoint)
    # for point in points:
    #     print(point.x, point.y)
    # print("----------------")


def expand(point, center_point_e, numc):
    global img_color  # point是未加入的点，center_point是此区域的中心点
    ka = 0.2
    kb = 0.4

    bi, gi, ri = img_color[point.y, point.x]

    bc, gc, rc = areargb(numc)

    # 强制转化成int类型，不然这些点的取值范围是0-255，会发生溢出
    bi = int(bi)
    gi = int(gi)
    ri = int(ri)
    bc = int(bc)
    gc = int(gc)
    rc = int(rc)
    t1 = math.sqrt(pow((bi - bc), 2) + pow((gi - gc), 2) + pow((ri - rc), 2))
    t2 = math.sqrt(pow((point.x - center_point_e.x), 2) + pow((point.y - center_point_e.y), 2))
    # if t1>160:
    #     t1 = 600
    t = ka * t1 + kb * t2

    t = math.ceil(t)
    # print("=============")
    # print(t1,t2)

    return t

def areargb(num):
    global points, r_list, g_list, b_list, center_point_total_num
    num -= 1

    bc = b_list[num] // center_point_total_num[num]
    gc = g_list[num] // center_point_total_num[num]
    rc = r_list[num] // center_point_total_num[num]

    return bc, gc, rc

# 函数：初始化的种子点打上标记
def seed_init(seed_points):
    global img_mark, center_point_totalx, center_point_total_num, center_point_totaly, img_color, r_list, b_list, g_list  # 声明全局变量
    i = 1

    for seed_point in seed_points:
        # 取的时候先x后y，用的时候先y后x
        img_mark[seed_point.y, seed_point.x] = i

        # 给中心点计算的和进行初始化
        center_point_totaly[i - 1] = center_point_totaly[i - 1] + seed_point.y
        center_point_totalx[i - 1] = center_point_totalx[i - 1] + seed_point.x
        center_point_total_num[i - 1] += 1

        # 计算rgb值
        b, g, r = img_color[seed_point.y, seed_point.x]
        b_list[i - 1] = b_list[i - 1] + b
        g_list[i - 1] = g_list[i - 1] + g
        r_list[i - 1] = r_list[i - 1] + r

        i = i + 1


if __name__ == '__main__':
    start_time = time.perf_counter()




    image = 'liunan.jpg'
    img = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
    img_color = cv2.imread(image,cv2.IMREAD_COLOR)
    img_k = img
    face_eg = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_eg.detectMultiScale(img)
    x = y = w = h = 0
    x1 = y1 = w1 = h1 = 0

    #x,y是左上角的点
    for(x,y,w,h) in faces:
        x1 = int(x+0.2*w)
        x2 = int(x + 0.8*w)
        y1 = int(y+0.23*h)
        y2 = int(y+0.50*h)
        # img_k = cv2.rectangle(img_k,(x1,y1),(x2,y2),(0,255,0),2)
        # print(x,y,w,h)
    x_num = x1
    y_num = y1
    w_num = 0.6 * w
    h_num = 0.27 * h
    #二值化了一下图像
    _, binary_image = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    #边缘检测
    edges = cv2.Canny(img, threshold1=220, threshold2=250)
    edges2 = cv2.Canny(binary_image, threshold1=230, threshold2=250)

    im_shape = img.shape
    height = im_shape[0]
    width = im_shape[1]

    #生成第一个加权矩阵
    img_weighting = np.zeros([height, width])
    weighting(edges,edges2,x1,y1,int(0.6*w),int(0.27*h))

    #惩罚函数矩阵
    img_expand = np.zeros([height, width])

    #已扩张点记录矩阵
    img_mark = np.zeros([height, width])

    # 中间记录矩阵
    img_mid = np.zeros([height, width])

    # 新建一个中间点列表，里边是打上惩罚函数但是未转化的点
    midpoints = []
    center_points = []

    center_point_totalx = [0] * (2)
    center_point_totaly = [0] * (2)
    # 装的是每一块，点的数目
    center_point_total_num = [0] * (2)
    # 新建一个rgb列表，用来快速计算rgb平均值，(n*n)代表初始化了n个0
    r_list = [0] * (2)
    g_list = [0] * (2)
    b_list = [0] * (2)


    # 定义膨胀操作的结构元素（核）
    kernel = np.ones((2, 2), np.uint8)

    # 应用膨胀操作
    # cv2.imshow('img33', edges)
    # cv2.imshow('img55', edges2)
    # cv2.imshow('img6', binary_image)
    # cv2.imshow('img2',dilated_image)
    # cv2.imshow('img',img_weighting)
    #
    # center_point(dilated_image,img_color)
    # cv2.imshow('img4', img_color)
    # cv2.imwrite('cc.jpg', img_weighting*255)
    # cv2.imshow('img1',img_weighting)
    close(x1,y1,int(0.6*w),int(0.27*h))


    # cv2.imshow('img2', img_weighting)

    n = 0
    while True:
        img_weighting = cv2.dilate(img_weighting, kernel, iterations=1)
        print(calculate_largest_components_percentage(img_weighting))
        n=n+1
        if calculate_largest_components_percentage(img_weighting) >80 or n>=10:
            break


    cv2.imshow('img3', img_weighting)
    xc1,yc1,xc2,yc2 = center_point(img_weighting, img_color)

    #初始化一下中心点
    center_points.append(Point(xc1,yc1))
    center_points.append(Point(xc2,yc2))

    #所有点的集合
    points = center_points
    for point in center_points:
        print(point.x, point.y)
    #把中心点作为种子点进行初始化
    seed_init(center_points)
    for i in range(160):

        change()

        # 2.更新区域中心点
        # 参数是上文中的分割平方数
        find_center_point()

        # 3.打惩罚函数

        grow()


    print("----------------")
    print("----------------")
    for point in center_points:
        print(point.x, point.y)
    print("----------------")
    print("----------------")



    # 给边界点划线
    for point in points:
        for i in range(8):
            tmpX = point.x + connects2[i].x
            tmpY = point.y + connects2[i].y
            if tmpX < 0 or tmpY < 0 or tmpY >= height or tmpX >= width:
                continue
            if img_mark[tmpY, tmpX] != img_mark[point.y, point.x]:
                img_color[point.y, point.x] = 0, 0, 255
                break
    cv2.imshow('img566', img_color)
    #窗口尺寸定义
    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('img', 650, 750)


    end_time = time.perf_counter()

    # 计算程序运行时间（以秒为单位）
    run_time = end_time - start_time

    print(f"程序运行时间为：{run_time}秒")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
