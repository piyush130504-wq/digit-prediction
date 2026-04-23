import cv2
import numpy as np
import tensorflow as tf


model = tf.keras.models.load_model('mnist_strong_cnn1.h5')


cap = cv2.VideoCapture(0)


canvas = np.zeros((480, 640), dtype=np.uint8)

drawing = False
prev_center = None
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    center = None
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 300:  # Filter noise
            x, y, w, h = cv2.boundingRect(largest)
            center = (x + w // 2, y + h // 2)
            cv2.circle(frame, center, 8, (0, 255, 0), -1)

            if drawing and prev_center:
                cv2.line(canvas, prev_center, center, 255, 8)
            prev_center = center
        else:
            prev_center = None
    else:
        prev_center = None

    
    overlay = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    combined = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

    
    cv2.putText(combined, "Press 'd' to draw, 'p' to predict, 'c' to clear, 'q' to quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)
    cv2.putText(combined, "Hold a red item in front of camera to draw in the air",
            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)

    cv2.imshow("Air-Drawing with Color Detection", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('d'):
        drawing = not drawing
        prev_center = None
    elif key == ord('c'):
        canvas.fill(0)
        prev_center = None
    elif key == ord('p'):
        x, y, w, h = cv2.boundingRect(canvas)
        if w > 10 and h > 10:
            roi = canvas[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (28, 28))
            roi_normalized = roi_resized / 255.0
            roi_reshaped = roi_normalized.reshape(1, 28, 28, 1)

            prediction = model.predict(roi_reshaped, verbose=0)
            digit = int(np.argmax(prediction))
            confidence = float(np.max(prediction)) * 100

            print(f"Predicted: {digit} ({confidence:.2f}%)")

            cv2.putText(combined, f"Predicted: {digit} ({confidence:.1f}%)",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.imshow("Air-Drawing with Color Detection", combined)
            cv2.waitKey(1000)
        else:
            print("No valid digit found. Try drawing again.")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
