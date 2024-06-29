import cv2

if __name__ == '__main__':
    img = cv2.imread('20du.jpg',cv2.IMREAD_GRAYSCALE)
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
    edges = cv2.Canny(img, threshold1=200, threshold2=250)
    edges2 = cv2.Canny(binary_image, threshold1=230, threshold2=250)

    im_shape = img.shape
    height = im_shape[0]
    width = im_shape[1]









    #窗口尺寸定义
    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('img', 650, 750)





    cv2.imshow('img',edges2)
    cv2.imshow('img2',edges)
    cv2.imshow('img3',binary_image)

    cv2.waitKey(0)
