# if only index finger is up mouse is scrolling# if two fingers (index & middle) in click mode, fingers must be close
# the camera
import cv2
# import numpy for graphs and stuff
import numpy as np
# get the hand tracking
from HandTrackingModule import HandTrackingModule as htm
# use the time module
import time

# importing os module
# import pyautogui to move mouse and use keyboard
# https://pyautogui.readthedocs.io/en/latest/quickstart.html
import pyautogui

# set width and height for the screen for the camera as variables
################################
wCam, hCam = 640, 480

frameReduction = 100  # Frame reduction rate
################################


# Image directory
directory = r'C:\Users\Sam\Desktop'

# values to smooth mouse movement
#number of seconds to wait
numSeconds = .02
#smoothening value
smooth = 5
# the previous location of the x and y regarding the mouse
plocX, plocY = 0, 0
# current location of mouse
clocX, clocY = 0, 0

# get video input
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# get the width of the screen

wScreen, hScreen = pyautogui.size()

# set time
pTime = 0
cTime = 0

# create detector object passing only one hand (for now )
detector = htm.handDetector(maxHands=1)

while True:
    # 1) find hand landmarks
    success, img = cap.read()
    # find the hands passing the image
    img = detector.findHands(img)

    # draw box around the hand passing the img
    lmList, bbox = detector.findPosition(img)

    # 2) Get tips of index and middle fingers
    # if the list is longer than 0, let's find the hand length
    if len(lmList) != 0:
        # the points of the index finger are x1, y1
        x1, y1 = lmList[8][1:]
        # middle finger is number 12
        x2, y2 = lmList[12][1:]
        # PRINT TO DEBUG
        # print(x1, y1, x2, y2)

        # 3) Check what fingers are up
        fingers = detector.fingersUp()
        # debug to print
        # print(fingers)
        # draw a rectangle that the user will be able to move the mouse in
        cv2.rectangle(img, (frameReduction, frameReduction), (wCam - frameReduction, hCam - frameReduction),
                      (255, 0, 255), 2)
        # 4) Only Index Finger? Moving Mode - move mouse
        # if the index finger is up and the middle finger is down
        if fingers[1] == 1 and fingers[2] == 0:
            # only measure in smaller rectangle to prevent tearing that happens at bottom of screen by adding in frameReduction
            # 5) convert coordinates to mouse location based on screen size and the reduced frameReduction,
            x3 = np.interp(x1, (frameReduction, wCam - frameReduction), (0, wScreen))

            y3 = np.interp(y1, (frameReduction, hCam - frameReduction), (0, hScreen))

            # 6) Smooth Values - reduces jitter and flicker do this
            # clocX = plocX + (x3 - plocX) / smooth
            # clocX = plocY + (y3 - plocY) / smooth
            # 7) Move Mouse over number of seconds
            # since the image is flipped subtract the wScreen from the x1 value
            pyautogui.moveTo(wScreen - x3, y3, numSeconds)
            # draw circle so we know that we are in moving mode
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            #plocX, plocY = clocX, clocY

        # 8) Check for Clicking Mode Both index and middle finger up? Clicking Mode time
        if fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 0 and fingers[3] == 0 and fingers[4] == 0:
            # if both fingers are up find the distance between the index finger (8) and the middle finger (12)
            # 9) find the distance between the index and middle finger
            length, img, lineInfo = detector.findDistance(8, 12, img)
            # debug to print
            # print(length)
            # darw a circle in green is length is less than 40
            if (length < 40):
                # draw circle so we know that we are in moving mode
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 255), cv2.FILLED)

                # 10) click mouse if distance is x close
                pyautogui.click()

        #take a picture with just the thumb
        if fingers[1] == 0 and fingers[2] == 0 and fingers[0] == 1 and fingers[3] == 0 and fingers[4] == 0:
            # Filename
            filename = 'savedImage.jpg'
            # Saving the image
            cv2.imwrite(filename, img)



    # 11) Frame Rate check
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
