import cv2
#print (cv2.__version__)
cap = cv2.VideoCapture(0)
w = 640
h = 480

while True:
    success, img = cap.read()
    img = cv2.resize(img,(w,h))
    cv2.imshow("Result",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break