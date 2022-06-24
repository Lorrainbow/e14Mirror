from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(640, 480))

def drawTriangle (eye, shape, image):
    if eye == "Left":
        start = 42
        end = 48
        tipPoint = 45
        top = 44
        bottom = 46
        color = (255,0,0)

        edge = 42
        edgeCoord = shape[edge]
        tipCoord = shape[tipPoint]
        change = int((tipCoord[0]-edgeCoord[0])/3)
        (x, y, w, h) = cv2.boundingRect(np.array([shape[edge:tipPoint]]))
        print("change is ",change)
        x=x-40
    else:
        start = 36
        end = 42
        tipPoint = 36
        top = 37
        bottom = 41
        edge = 39

        edgeCoord = shape[edge]
        tipCoord = shape[tipPoint]
        change = int((tipCoord[0]-edgeCoord[0])/3)
        #cv2.circle(image, (shape[tipPoint]), 1, (255,255,0), -1)
        color = (0,255,0)
        (x, y, w, h) = cv2.boundingRect(np.array([shape[tipPoint:edge]]))
        x=x-100

    w =250
    h =100
    y=y-20

    cropped = image[shape[tipPoint][0]:shape[tipPoint][0] - h, shape[top][1]:shape[bottom][1] - w]
    #for (x, y) in shape[start:end]:
        #cv2.circle(image, (x, y), 1, color, -1)
    
    #take the corner eye coordinate
    #make the change to the tip of our triangle
    tip = shape[tipPoint]
    tipX =tip[0]
    tipY =tip[1]
    triangleTip = (tipX+change,tipY)
    
    print("tip: ",shape[tipPoint])
    print("top: ",shape[edge])
    
    #circle our triangletip to check 
#    cv2.circle(image, triangleTip, 1, (0, 255, 0), -1)
#    cv2.circle(image, shape[top],  1, (0, 255, 0), -1)
#    cv2.circle(image, shape[bottom], 1, (0,255,0),-1)
    
    #create an array of points
    pts=[shape[top],triangleTip,shape[bottom]]
    pts=np.array(pts)
#    pts = np.array([top,tipPoint,bottom], np.int32)            
    pts = pts.reshape((-1, 1, 2))      
    isClosed = False
    
    #add the triangle to the image
    image = cv2.polylines(image, [pts], isClosed, (255,0,0), 1)
    
    return image

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
help="facial landmark predictor path")
args = vars(ap.parse_args())
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    image = imutils.resize(image, width=600)
    rawCapture.truncate(0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

# loop over the face detections
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)        

        image = drawTriangle("Left", shape, image)
        image = drawTriangle("Right", shape, image)
        
            
    cv2.imshow("Image",image)

    if cv2.waitKey(1) == 27:
        break
