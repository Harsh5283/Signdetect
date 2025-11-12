import cv2

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print("Camera not found")
        break

    cv2.imshow("Press S to Save, Q to Quit", img)

    key = cv2.waitKey(1)

    if key == ord('s'):  # press s to save image
        cv2.imwrite("captured_image.jpg", img)
        print("Image Saved as captured_image.jpg")

    if key == ord('q'):  # press q to quit
        break

cap.release()
cv2.destroyAllWindows()
