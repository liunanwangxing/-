import cv2
import numpy as np

def calculate_largest_components_percentage(image_path):
    # 读取图像并转换为灰度图像
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read image")
        return

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化处理
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

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

    # 输出结果
    print(f"Total connected components: {num_labels - 1}")
    print(f"Largest component area: {largest_areas[0][0]}")
    print(f"Second largest component area: {largest_areas[1][0]}")
    print(f"Total area of all components: {total_area}")
    print(f"Percentage of the two largest components: {largest_percentage:.2f}%")

    # 可视化结果并标注连通域
    output_image = np.zeros_like(image)
    for i, (area, label) in enumerate(component_areas, start=1):
        color = (0, 255, 0) if (area, label) in largest_areas else (0, 0, 255)
        output_image[labels_im == label] = color

        # 找到连通域的质心以放置标签
        moments = cv2.moments((labels_im == label).astype(np.uint8))
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
        cv2.putText(output_image, str(i), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('Labeled Connected Components', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 使用函数
calculate_largest_components_percentage('juji2.jpg')
