import cv2
import numpy as np

# 读取图像并转换为灰度图像
image = cv2.imread('cc.jpg')

# 检查图像是否成功读取
if image is None:
    print("Error: Could not read image")
    exit()

# 转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 二值化处理
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# 定义闭运算的结构元素（核）
kernel = np.ones((5, 5), np.uint8)

# 应用闭运算
closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
closed_image2 = cv2.morphologyEx(closed_image, cv2.MORPH_CLOSE, kernel)
closed_image3 = cv2.morphologyEx(closed_image2, cv2.MORPH_CLOSE, kernel)
closed_image4 = cv2.morphologyEx(closed_image3, cv2.MORPH_CLOSE, kernel)

# 显示原始图像和闭运算后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Binary Image', binary_image)
cv2.imshow('Closed Image', closed_image)
cv2.imshow('Closed Image2', closed_image2)
cv2.imshow('Closed Image3', closed_image3)
cv2.imshow('Closed Image4', closed_image4)

# 保存闭运算后的图像
# cv2.imwrite('closed_image.jpg', closed_image)

# 等待用户按键
cv2.waitKey(0)

# 关闭所有窗口
cv2.destroyAllWindows()
