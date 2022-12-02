import random
import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import math
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
 
detector = HandDetector(detectionCon=0.8, maxHands=1)
 
 
class SnakeGameClass:
    def __init__(self, pathFood):
        self.points = []  # all points of the snake
        self.lengths = []  # distance between each point
        self.currentLength = 0  # total length of the snake
        self.allowedLength = 150  # total allowed Length
        self.previousHead = 0, 0  # previous head point ( x , y )
 
        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED) # read the food image
        self.hFood, self.wFood, _ = self.imgFood.shape # get its shapes
        self.foodPoint = 0, 0 # initialize  the location of the food
        self.randomFoodLocation() # then randomize it using the randomFoodLocation() function
 
        self.score = 0 # your current scor
        self.gameOver = False
 
    def randomFoodLocation(self):
        # new random food location : 
            # x : random int from 100 px to 1000 px
            # y : random int from 100 px to 600 px
        self.foodPoint = random.randint(100, 1000), random.randint(100, 600)
 
    def update(self, imgMain, currentHead):
 
        if self.gameOver:
            cvzone.putTextRect(imgMain, "Game Over", [300, 400],
                               scale=7, thickness=5, offset=20)
            cvzone.putTextRect(imgMain, f'Your Score: {self.score}', [300, 550],
                               scale=7, thickness=5, offset=20)
        else:
            px, py = self.previousHead # previous head coordinates
            cx, cy = currentHead # current head coordinates
            # add the new head to the points list 
            self.points.append([cx, cy]) 
            # calculate te distance between the new and the prevoius head 
            # and append it to the list
            distance = math.hypot(cx - px, cy - py)
            self.lengths.append(distance)
            # add distance to current length
            self.currentLength += distance
            # update the previous head
            self.previousHead = cx, cy
 
            # Length Reduction
            if self.currentLength > self.allowedLength:
                for i, length in enumerate(self.lengths):
                    self.currentLength -= length
                    self.lengths.pop(i)
                    self.points.pop(i)
                    if self.currentLength < self.allowedLength:
                        break
 
            # Check if snake ate the Food
            rx, ry = self.foodPoint
            
            if rx - self.wFood // 2 < cx < rx + self.wFood // 2 and \
                    ry - self.hFood // 2 < cy < ry + self.hFood // 2:
                # if the current head is near the food
                # 1 - change the location of the food
                self.randomFoodLocation()
                # 2 - change the length
                self.allowedLength += 50
                # 3 - add 1 to score
                self.score += 1
                print(self.score)
 
            # Draw Snake
            if self.points:
                # draw the lines 
                for i, point in enumerate(self.points):
                    if i != 0:
                        cv2.line(imgMain, self.points[i - 1], self.points[i], (0, 0, 255), 20)
                # draw the head of the snake
                cv2.circle(imgMain, self.points[-1], 20, (0, 255, 0), cv2.FILLED)
 
            # Draw Food
            imgMain = cvzone.overlayPNG(imgMain, self.imgFood,
                                        (rx - self.wFood // 2, ry - self.hFood // 2))
 
            cvzone.putTextRect(imgMain, f'Score: {self.score}', [50, 80],
                               scale=3, thickness=3, offset=10)
 
            # Check for Collision
            pts = np.array(self.points[:-2], np.int32) # take all the points - the last 2 (head and the previous head)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(imgMain, [pts], False, (0, 255, 0), 3)
            #compute distance between head and each point
            minDist = cv2.pointPolygonTest(pts, (cx, cy), True)
            # if any distance i below a threshold ( 1 or - 1) Collision detected !
            if -1 <= minDist <= 1:
                print("Hit")
                self.gameOver = True
                self.points = []  # all points of the snake
                self.lengths = []  # distance between each point
                self.currentLength = 0  # total length of the snake
                self.allowedLength = 150  # total allowed Length
                self.previousHead = 0, 0  # previous head point
                self.randomFoodLocation()
 
        return imgMain
 
 
game = SnakeGameClass("Donut.png")
 
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)
 
    if hands:
        lmList = hands[0]['lmList']
        pointIndex = lmList[8][0:2]
        img = game.update(img, pointIndex)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        game.gameOver = False
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()