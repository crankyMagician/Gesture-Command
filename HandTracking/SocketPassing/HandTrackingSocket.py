from HandTrackingModule import handDetector
import cv2
import socket

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
#reading the image
success, img = cap.read()
h, w, _ = img.shape
#the hand detector
detector = handDetector(detectionCon=0.8, maxHands=2)

#the socket to open
socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#opening port 5052
serverAddressPort = ("127.0.0.1", 5052)

while True:
    # Get image frame
    success, img = cap.read()
    # Find the hand and its landmarks
    hands, img = detector.findHands(img)  # with draw
    # hands = detector.findHands(img, draw=False)  # without draw
    dataHandOne = []
    #the second hand
    dataHandTwo = []
    #if there are hands
    if hands:
        # Hand 1
        hand = hands[0]
        lmList = hand["lmList"]  # List of 21 Landmark points
        for lm in lmList:
            dataHandOne.extend([lm[0], h - lm[1], lm[2]])

        socket.sendto(str.encode(str(dataHandOne)), serverAddressPort)
        #add a second hand?
        if len(hands) == 2:
            # Hand 2
            hand2 = hands[1]
            lmList2 = hand2["lmList"]  # List of 21 Landmark points
            for lm in lmList2:
                dataHandTwo.extend([lm[0], h - lm[1], lm[2]])
            socket.sendto(str.encode(str(dataHandTwo)), serverAddressPort)

    # Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
