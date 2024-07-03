import time
import heapq
import cv2
import numpy as np

def close(x,y,w,h):
    global imgg
    for i in range(w):
        for j in range(h):
            if imgg[y + j, x + i] == 1:
                continue
            s1 = s2 = s3 = s4 = 0
            for k in range(10):
                # if y + j + k >h-10 or y + j - k < 10 or x + i + k>w-10 or x + i - k < 10:
                #     continue
                if imgg[y + j + k, x + i] == 1:
                    s1 = 1
                    print(1)
                if imgg[y + j - k, x + i] == 1:
                    s2 = 1
                    print(2)
                if imgg[y + j, x + i + k] == 1:
                    s3 = 1
                    print(3)
                if imgg[y + j, x + i - k] == 1:
                    s4 = 1
                    print(4)
            #
            if s1+s2+s3+s4 >= 4:
                imgg[y + j, x + i] = 1
                print(1)
if __name__ == '__main__':
    start_time = time.perf_counter()
    image = '9.png'
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img_color = cv2.imread(image, cv2.IMREAD_COLOR)
    im_shape = img.shape
    height = im_shape[0]
    width = im_shape[1]
    print(height, width)

    _, binary_image = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    imgg = binary_image
    cv2.imshow('img1', imgg)
    close(0, 0, width, height)
    cv2.imshow('img2', imgg)
    end_time = time.perf_counter()

    # 计算程序运行时间（以秒为单位）
    run_time = end_time - start_time

    print(f"程序运行时间为：{run_time}秒")

    cv2.waitKey(0)
    cv2.destroyAllWindows()