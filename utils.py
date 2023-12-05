import cv2

def showImage(img, scale_factor=1, padding=50):

    img = cv2.resize(img, (0,0), fx=scale_factor, fy=scale_factor)
    cv2.namedWindow("Chip Image", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Chip Image", img)
    cv2.resizeWindow("Chip Image", int(img.shape[0] * scale_factor) + padding, int(img.shape[1] * scale_factor) + padding)
    cv2.waitKey(0)
