import cv2
import numpy as np

# 读取图像
image = cv2.imread('juji2.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('down.jpg', cv2.IMREAD_COLOR)

# 二值化处理
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# 寻找轮廓
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 计算每个轮廓的面积，并保留最大的两个轮廓
contour_areas = [(cv2.contourArea(contour), contour) for contour in contours]
contour_areas.sort(key=lambda x: x[0], reverse=True)
largest_contours = [contour_areas[i][1] for i in range(min(2, len(contour_areas)))]

# 创建彩色图像以绘制结果
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

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

    # 在输出图像上绘制质心
    cv2.circle(image2, (cX, cY), 1, (0, 0, 255), -1)
    cv2.putText(image2, f"({cX}, {cY})", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# 显示结果
cv2.imshow('Detected Centers', image2)

# 保存结果图像
cv2.imwrite('detected_centers.jpg', image2)

# 等待用户按键
cv2.waitKey(0)

# 关闭所有窗口
cv2.destroyAllWindows()

# 打印每个质心的坐标
for contour in largest_contours:
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    print(f"Center: ({cX}, {cY})")
