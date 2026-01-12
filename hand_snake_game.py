import cv2
import mediapipe as mp
import numpy as np
import random

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

WIDTH, HEIGHT = 640, 480

# Snake
snake = [(320, 240)]
snake_len = 1
smooth_factor = 0.25   # smooth slow follow

# Food
food_x = random.randint(50, WIDTH-50)
food_y = random.randint(50, HEIGHT-50)
score = 0
game_over = False

def reset_game():
    global snake, snake_len, score, food_x, food_y, game_over
    snake = [(320,240)]
    snake_len = 1
    score = 0
    food_x = random.randint(50, WIDTH-50)
    food_y = random.randint(50, HEIGHT-50)
    game_over = False

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (WIDTH, HEIGHT))

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not game_over and result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark

        fx = int(lm[8].x * WIDTH)
        fy = int(lm[8].y * HEIGHT)

        hx, hy = snake[0]

        new_x = int(hx + (fx - hx) * smooth_factor)
        new_y = int(hy + (fy - hy) * smooth_factor)

        snake.insert(0, (new_x, new_y))
        if len(snake) > snake_len:
            snake.pop()

        # Wall collision
        if new_x <= 10 or new_x >= WIDTH-10 or new_y <= 10 or new_y >= HEIGHT-10:
            game_over = True

        # Food collision
        if abs(new_x - food_x) < 15 and abs(new_y - food_y) < 15:
            food_x = random.randint(50, WIDTH-50)
            food_y = random.randint(50, HEIGHT-50)
            snake_len += 1
            score += 1

        draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

    # Draw snake
    for p in snake:
        cv2.circle(img, p, 8, (0,255,0), -1)

    # Food
    cv2.circle(img, (food_x, food_y), 10, (0,0,255), -1)

    # Score panel
    cv2.rectangle(img, (0,0), (WIDTH,40), (0,0,0), -1)
    cv2.putText(img, f"Score: {score}", (15,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    if game_over:
        cv2.rectangle(img, (0,200), (WIDTH,280), (0,0,0), -1)
        cv2.putText(img, "GAME OVER", (180,240),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,0,255), 3)
        cv2.putText(img, "Press R to Restart", (160,270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("AI Finger Controlled Snake Game", img)

    key = cv2.waitKey(1)
    if key == ord('r'):
        reset_game()
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
