import cv2
import cvzone
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Hand detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Colors
colorR = (19, 69, 139)  # Brown color (BGR)
cornerColor = (0, 0, 0)  # Black for corners
alpha = 0.6  # Transparency level (60%)

# Class for draggable and resizable boxes
class DragRect():
    def __init__(self, posCenter, size=[200, 200]):
        self.posCenter = list(posCenter)  # Tuple changed to list
        self.size = list(size)  # Tuple changed to list
        self.dragging = False
        self.resizing = False
        self.resize_edge = None
        self.start_cursor = (0, 0)
        self.start_size = (0, 0)
        self.start_pos = (0, 0)

    def update(self, cursor, isPinching):
        cx, cy = self.posCenter
        w, h = self.size
        margin = 20  # Edge/corner detection margin

        if isPinching:
            if not self.resizing and not self.dragging:
                # Check corners
                corners = [
                    (cx - w//2, cy - h//2),  # Top-left
                    (cx + w//2, cy - h//2),  # Top-right
                    (cx - w//2, cy + h//2),  # Bottom-left
                    (cx + w//2, cy + h//2)   # Bottom-right
                ]
                for i, (x, y) in enumerate(corners):
                    if abs(cursor[0] - x) < margin and abs(cursor[1] - y) < margin:
                        self.resizing = True
                        self.resize_edge = ['top-left', 'top-right', 'bottom-left', 'bottom-right'][i]
                        self.start_cursor = cursor
                        self.start_size = list(self.size)
                        self.start_pos = list(self.posCenter)
                        return

                # Check edges
                if (abs(cursor[0] - (cx - w//2)) < margin) and (cy - h//2 < cursor[1] < cy + h//2):
                    self.resizing = True
                    self.resize_edge = 'left'
                elif (abs(cursor[0] - (cx + w//2)) < margin) and (cy - h//2 < cursor[1] < cy + h//2):
                    self.resizing = True
                    self.resize_edge = 'right'
                elif (abs(cursor[1] - (cy - h//2)) < margin) and (cx - w//2 < cursor[0] < cx + w//2):
                    self.resizing = True
                    self.resize_edge = 'top'
                elif (abs(cursor[1] - (cy + h//2)) < margin) and (cx - w//2 < cursor[0] < cx + w//2):
                    self.resizing = True
                    self.resize_edge = 'bottom'

                if self.resizing:
                    self.start_cursor = cursor
                    self.start_size = list(self.size)
                    self.start_pos = list(self.posCenter)
                    return

                # Check for dragging
                if (cx - w//2 < cursor[0] < cx + w//2) and (cy - h//2 < cursor[1] < cy + h//2):
                    self.dragging = True
                    self.posCenter = list(cursor)
            else:
                if self.resizing:
                    delta_x = cursor[0] - self.start_cursor[0]
                    delta_y = cursor[1] - self.start_cursor[1]
                    min_size = 50  # Minimum box size

                    if self.resize_edge == 'right':
                        new_width = max(min_size, self.start_size[0] + delta_x)
                        self.size[0] = new_width
                    elif self.resize_edge == 'left':
                        new_width = max(min_size, self.start_size[0] - delta_x)
                        self.size[0] = new_width
                        self.posCenter[0] = self.start_pos[0] + delta_x / 2
                    elif self.resize_edge == 'top':
                        new_height = max(min_size, self.start_size[1] - delta_y)
                        self.size[1] = new_height
                        self.posCenter[1] = self.start_pos[1] + delta_y / 2
                    elif self.resize_edge == 'bottom':
                        new_height = max(min_size, self.start_size[1] + delta_y)
                        self.size[1] = new_height

                elif self.dragging:
                    self.posCenter = list(cursor)
        else:
            self.dragging = False
            self.resizing = False
            self.resize_edge = None

# Start with 3 boxes
rectList = [DragRect([x * 300 + 200, 200]) for x in range(3)]

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgOverlay = img.copy()  # Copy of the image for transparency effect

    hands, img = detector.findHands(img, flipType=False)

    if hands:
        for hand in hands:
            lmList = hand["lmList"]
            if len(lmList) > 12:
                x1, y1 = lmList[8][:2]  # Index finger
                x2, y2 = lmList[12][:2]  # Middle finger
                pinch_dist = detector.findDistance((x1, y1), (x2, y2))[0]
                isPinching = pinch_dist < 50

                for rect in reversed(rectList):
                    rect.update((x1, y1), isPinching)
                    if rect.dragging or rect.resizing:
                        break  # Stop after finding first interactive box

            # Detect fist (all fingers folded)
            fingers = detector.fingersUp(hand)
            if fingers == [0, 0, 0, 0, 0]:  # Fist detected
                new_box = DragRect([np.random.randint(200, 1080), np.random.randint(200, 500)])
                rectList.append(new_box)

    # Draw semi-transparent boxes
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size

        # Create a transparent overlay
        overlay = img.copy()
        cv2.rectangle(overlay, (int(cx - w/2), int(cy - h/2)), (int(cx + w/2), int(cy + h/2)), colorR, cv2.FILLED)
        cvzone.cornerRect(overlay, (int(cx - w/2), int(cy - h/2), int(w), int(h)), 20, rt=0, colorC=cornerColor)

        # Blend overlay with the image
        cv2.addWeighted(overlay, alpha, imgOverlay, 1 - alpha, 0, imgOverlay)

    # Show the final blended image
    cv2.imshow("Image", imgOverlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
