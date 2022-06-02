# import the hand tracking
import math

import cv2
# get media pipe
import mediapipe as mp
# get time to check frame rate
import time


# create a class that correlates to hand class
# create max hands class
# create detection confidence
class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.lmList = None
        self.results = None
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, 2,
                                        self.detectionCon, self.trackCon)

        # draw
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        # convert it to the rgb scope
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # process the results for hands
        self.results = self.hands.process(imgRGB)
        # print the results if a hand is recognized
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:

            for handLMS in self.results.multi_hand_landmarks:
                if draw:
                    # draw maps with media pipe solution
                    self.mpDraw.draw_landmarks(img, handLMS, self.mpHands.HAND_CONNECTIONS)

        # return image with drawing
        return img

    # find the hand's position
    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                              (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img, draw=True):

        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    # current time
    cTime = 0
    # previous time
    pTime = 0

    # capture video
    cap = cv2.VideoCapture(0)

    detector = handDetector()

    while True:
        # if success capture and read
        success, img = cap.read()
        # on success set img to returned img from detector
        img = detector.findHands(img)

        # get lm list passing img
        lmList = detector.findPosition(img)

        if len(lmList) != 0:
            print(lmList[0])

        # simple frame calculator
        cTime = time.time()
        # calc fps 1/current time - previous time
        fps = 1 / (cTime - pTime)
        # previous time is now current time
        pTime = cTime

        # display it on screen, cast fps to int so no decimals, position, font, scale, and thickness
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                    3, (240, 248, 255), 2)

        # show the image
        cv2.imshow("Image", img)
        # give stuff a second to load
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
