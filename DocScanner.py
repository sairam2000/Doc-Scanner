import cv2
import numpy as np

org_img = cv2.imread("phy.jpg")
gray = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 70, 260)
img, counters, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
print(len(counters))
sort = sorted(counters, key=cv2.contourArea, reverse=True)[:5]
for c in sort:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*peri, True)
    if len(approx) == 4:
        cnt = approx
        break
img = cv2.drawContours(org_img.copy(), [cnt], -1, (0, 255, 255), 5)
dst = np.float32([[[500, 0]], [[0, 0]], [[0, 800]], [[500, 800]]])
t = cv2.getPerspectiveTransform(np.float32(cnt), dst)
img = cv2.warpPerspective(img, t, (500, 800))
# _, img = cv2.threshold(img, 50, 250, cv2.THRESH_TOZERO)
cv2.imshow("After Scanning", img)
cv2.imshow("Original Image", org_img)
cv2.imwrite("After Scanning.jpg", img)
cv2.imwrite("Original Image.jpg", org_img)
if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()

