import cv2
import numpy as np
import time
import math
from pydobot.dobot import MODE_PTP
from ultralytics import YOLO
from pydobot.dobot import MODE_PTP

# ---- Offsets and Parameters for Grid, Camera Location, and EndEffector ----
gCenter = [255, 0]
cameraLocation = [200, 0, 70] ## location of where the camera takes a picture
dz = -13 #delta Z for offsetting the height of the pen
gWidth = 20 # adjust for the size of your desired grid
gHeight = (gWidth*3)//2 
block1 = [gCenter[0]-gWidth, gCenter[1]-gWidth]
block2 = [gCenter[0]-gWidth, gCenter[1]]
block3 = [gCenter[0]-gWidth, gCenter[1]+gWidth]
block4 = [gCenter[0], gCenter[1]-gWidth]
block5 = [gCenter[0], gCenter[1]]
block6 = [gCenter[0], gCenter[1]+gWidth]
block7 = [gCenter[0]+gWidth, gCenter[1]-gWidth]
block8 = [gCenter[0]+gWidth, gCenter[1]]
block9 = [gCenter[0]+gWidth, gCenter[1]+gWidth]

corner1 = [gCenter[0]-gHeight, gCenter[1]-gHeight]
corner2 = [gCenter[0]-gHeight, gCenter[1]+gHeight]
corner3 = [gCenter[0]+gHeight, gCenter[1]+gHeight]
corner4 = [gCenter[0]+gHeight, gCenter[1]-gHeight]

cellPosition = {
    (0,0):block1,(0,1):block2,(0,2):block3,
    (1,0):block4,(1,1):block5,(1,2):block6,
    (2,0):block7,(2,1):block8,(2,2):block9
}

regionOfInterest = None
confidenceValue = 0.8
model = YOLO("XandOWieghts.pt")

# ---------------- Game Logic ----------------
def checkWinner(board):
    winLines = [
        [(0,0),(0,1),(0,2)], [(1,0),(1,1),(1,2)], [(2,0),(2,1),(2,2)],
        [(0,0),(1,0),(2,0)], [(0,1),(1,1),(2,1)], [(0,2),(1,2),(2,2)],
        [(0,0),(1,1),(2,2)], [(0,2),(1,1),(2,0)]
    ]
    for line in winLines:
        cells = [board[r][c] for r,c in line]
        if cells[0] != " " and cells[0] == cells[1] == cells[2]:
            return cells[0], line
    if all(cell != " " for row in board for cell in row):
        return "Drawn", None
    return None, None


def minimax(board, depth, isMaximizing, robotSymbol, playerSymbol):
    winner, _ = checkWinner(board)
    if winner == robotSymbol:
        return 10 - depth
    elif winner == playerSymbol:
        return depth - 10
    elif winner == "Drawn":
        return 0

    if isMaximizing:
        best = -float("inf")
        for r in range(3):
            for c in range(3):
                if board[r][c] == " ":
                    board[r][c] = robotSymbol
                    score = minimax(board, depth + 1, False, robotSymbol, playerSymbol)
                    board[r][c] = " "
                    best = max(best, score)
        return best
    else:
        best = float("inf")
        for r in range(3):
            for c in range(3):
                if board[r][c] == " ":
                    board[r][c] = playerSymbol
                    score = minimax(board, depth + 1, True, robotSymbol, playerSymbol)
                    board[r][c] = " "
                    best = min(best, score)
        return best


def chooseRobotMove(board, robotSymbol="O", playerSymbol="X"):
    priority = [(1,1),(0,0),(0,2),(2,0),(2,2),(0,1),(1,0),(1,2),(2,1)]

    for r,c in priority:
        if board[r][c] == " ":
            board[r][c] = robotSymbol
            w, _ = checkWinner(board)
            board[r][c] = " "
            if w == robotSymbol:
                return (r,c)

    for r,c in priority:
        if board[r][c] == " ":
            board[r][c] = playerSymbol
            w, _ = checkWinner(board)
            board[r][c] = " "
            if w == playerSymbol:
                return (r,c)

    best_score = -float("inf")
    best_move = None
    for r,c in priority:
        if board[r][c] == " ":
            board[r][c] = robotSymbol
            score = minimax(board, 0, False, robotSymbol, playerSymbol)
            board[r][c] = " "
            if score > best_score:
                best_score = score
                best_move = (r,c)
    return best_move


# ---------------- Vision ----------------
def detectGrid(frame):
    global regionOfInterest
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Filters for better vision stability
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    edges = cv2.Canny(blur, 40, 100)

    if regionOfInterest is None:
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 100, 20)
        if lines is None:
            return [[" "]*3 for _ in range(3)], frame
        horiz, vert = [], []
        for x1,y1,x2,y2 in lines[:,0]:
            if abs(y1 - y2) < 20: horiz += [y1, y2]
            elif abs(x1 - x2) < 20: vert += [x1, x2]
        if len(horiz) < 2 or len(vert) < 2:
            return [[" "]*3 for _ in range(3)], frame
        top, bottom, left, right = min(horiz), max(horiz), min(vert), max(vert)
        regionOfInterest = (top, bottom, left, right)

    top, bottom, left, right = regionOfInterest
    roi = frame[top:bottom, left:right]
    if roi.size == 0:
        return [[" "]*3 for _ in range(3)], frame

    results = model.predict(roi, conf=confidenceValue, verbose=False, device="cpu")
    detections = results[0].boxes.data.cpu().numpy()
    h, w, _ = roi.shape
    cHeight, cWidth = h // 3, w // 3
    new_board = [[" "]*3 for _ in range(3)]

    for det in detections:
        x1,y1,x2,y2,conf,cls = det
        cls = int(cls)
        if cls == 0: continue
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        row, col = 2 - (cy // cHeight), 2 - (cx // cWidth)
        if 0 <= row < 3 and 0 <= col < 3:
            new_board[row][col] = "O" if cls == 1 else "X"

    return new_board, frame


# ---------------- Dobot Functions ----------------
def draw(device, block, robotSymbol): ## Draws X or O on a designated spot
    callArray = [block1,block2,block3,block4,block5,block6,block7,block8,block9]
    idx = block - 1
    if not (0 <= idx < 9):
        return
    pos = callArray[idx]

    if robotSymbol == "X":
        drawX(device, pos[0], pos[1])
    else:
        drawO(device, pos[0], pos[1])



def drawX(device, x0, y0, size=15, zDraw=dz, zLift=dz+20): ## Draws X with a size of 15mm diagonal
    """Draws a clean X centered at (x0, y0)."""
    half = size / 2

    # Stroke 1: bottom-left → top-right
    device.move_to(mode=int(MODE_PTP.MOVJ_XYZ), x=x0 - half, y=y0 - half, z=zLift, r=0)  # pen up
    device.move_to(mode=int(MODE_PTP.MOVJ_XYZ), x=x0 - half, y=y0 - half, z=zDraw, r=0)  # pen down
    device.move_to(mode=int(MODE_PTP.MOVJ_XYZ), x=x0 + half, y=y0 + half, z=zDraw, r=0)  # draw line
    device.move_to(mode=int(MODE_PTP.MOVJ_XYZ), x=x0 + half, y=y0 + half, z=zLift, r=0)  # pen up

    # Stroke 2: top-left → bottom-right
    device.move_to(mode=int(MODE_PTP.MOVJ_XYZ), x=x0 - half, y=y0 + half, z=zLift, r=0)
    device.move_to(mode=int(MODE_PTP.MOVJ_XYZ), x=x0 - half, y=y0 + half, z=zDraw, r=0)
    device.move_to(mode=int(MODE_PTP.MOVJ_XYZ), x=x0 + half, y=y0 - half, z=zDraw, r=0)
    device.move_to(mode=int(MODE_PTP.MOVJ_XYZ), x=x0 + half, y=y0 - half, z=zLift, r=0)

    # Return to camera position
    device.move_to(mode=int(MODE_PTP.MOVJ_XYZ),
                   x=cameraLocation[0], y=cameraLocation[1],
                   z=cameraLocation[2], r=0)


def drawO(device, x0, y0, radius=5, z=dz, segments=72): ## Draws a smooth O shape using math with a radius of 5mm
    points = [
        (x0 + radius * math.cos(2 * math.pi * i / segments),
         y0 + radius * math.sin(2 * math.pi * i / segments))
        for i in range(segments + 1)
    ]

    for x, y in points:
        device.move_to(mode=int(MODE_PTP.MOVJ_XYZ), x=x, y=y, z=z, r=0)

    # Return to the camera position
    device.move_to(
        mode=int(MODE_PTP.MOVJ_XYZ),
        x=cameraLocation[0],
        y=cameraLocation[1],
        z=cameraLocation[2],
        r=0
    )

def printGameState(board):
    print("\nCurrent Game State:")
    for i in range(3):
        print(" | ".join(board[i]))
        if i < 2:
            print("-" * 9)
    print()

