
import cv2
import time
import threading
import pydobot
from serial.tools import list_ports
from Functions import *

# ---------------- Dobot Setup and Connection ----------------
available_ports = list_ports.comports()
dobot_port = next((p.device for p in available_ports if "usb" in p.device or "COM" in p.device), None)
if not dobot_port:
    raise Exception("No Dobot detected. Check your connection.")
print(f"Connected to Dobot on {dobot_port}")
device = pydobot.Dobot(port=dobot_port)

# ---------------- Game Setup ----------------
cap = cv2.VideoCapture(0) ##change for desired webcam
grid = [[" "]*3 for _ in range(3)]
previousGrid = None
playerSymbol = None
robotSymbol = None
turn = None
playerMoved = True

choice = input("Who is first, human or robot? (H=Human, R=Robot): ").strip().upper()
if choice == "H":
    print("Human player is first")
    turn = "HUMAN"
else:
    print("Robot is first and is X")
    robotSymbol = "X"
    playerSymbol = "O"
    turn = "Robot"

# ---------------- Main Game Loop ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    detectedGrid, vis_frame = detectGrid(frame)

    cv2.imshow("Tic-Tac-Toe Grid", vis_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # Detect First Player Move
    if choice == "H" and playerSymbol is None:
        first_moves = [(r,c,detectedGrid[r][c]) for r in range(3) for c in range(3) if detectedGrid[r][c] != " "]
        if first_moves:
            playerSymbol = first_moves[0][2]
            robotSymbol = "O" if playerSymbol == "X" else "X"
            grid = detectedGrid
            print(f"Player is {playerSymbol}, Robot is {robotSymbol}")
            printGameState(grid)
            turn = "Robot"

    # Check for winner
    winner, line = checkWinner(detectedGrid)
    if winner:
        if winner in ["X", "O"]:
            print(f"{winner} wins!")
        else:
            print("The game is a Draw!")
        break

    # Human-player turn
    if turn == "HUMAN":
        diffs = [(r,c,detectedGrid[r][c]) for r in range(3) for c in range(3)
                 if grid[r][c] == " " and detectedGrid[r][c] != " "]
        if diffs:
            r,c,symbol = diffs[0]
            if symbol == playerSymbol:
                grid[r][c] = symbol
                print(f"Player moved at ({r},{c})")
                printGameState(grid)
                turn = "Robot"

    # Robot turn
    elif turn == "Robot":
        move = chooseRobotMove(grid, robotSymbol, playerSymbol)
        if move:
            r,c = move
            grid[r][c] = robotSymbol
            print(f"Robot moves at ({r},{c})")
            printGameState(grid)
            threading.Thread(target=draw, args=(device, r*3 + c + 1, robotSymbol)).start()
            turn = "HUMAN"

cap.release()
cv2.destroyAllWindows()
