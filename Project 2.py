import cv2
import numpy as np
widthImg= 640
heightImg = 480
vid = cv2.VideoCapture(0)
vid.set(3, widthImg)
vid.set(4, heightImg)
vid.set(10,150)

def preProcessor(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=2)
    imgThres = cv2.erode(imgDial,kernel,iterations=1)

    return imgThres


def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            cv2.drawContours(imgCount,cnt,-1,(0,0,255), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    return biggest

def reOrder(myPoints):
    myPoints = myPoints.reshape((4,2))
    newPoints = np.zeros((4,1,2), np.int32)
    add = myPoints.sum(1)
    newPoints[0] = myPoints[np.argmin(add)]
    newPoints[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, 1)
    newPoints[1] = myPoints[np.argmin(diff)]
    newPoints[2] = myPoints[np.argmax(diff)]
    return newPoints

def warper(img, biggest):
    biggest = reOrder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0], [widthImg,0], [0, heightImg], [widthImg,heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    return imgOutput

while True:
    succ, img = vid.read()
    img = cv2.resize(img, (widthImg,heightImg))
    imgCount = img.copy()
    imgThres = preProcessor(img)
    biggest = getContours(imgThres)
    imgWarped = warper(img,biggest)
    cv2.imshow("Result", imgWarped)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break