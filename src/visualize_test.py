import cv2

image_path = "/home/quangmanh/Documents/pineline/back-end-label/uploads/images/classes-van-phong-pham/Giấy/0b223c80-6094-4719-955b-43efd3d709f3.jpg"
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(image_path)

x1, y1, x2, y2, conf = 6.310701, 156.798935, 741.324951, 587.981995, 0.5665
label = "bed"

pt1 = (int(x1), int(y1))
pt2 = (int(x2), int(y2))
cv2.rectangle(img, pt1, pt2, (0,255,0), 2)
cv2.putText(img, f"{label} {conf:.2f}", (pt1[0], pt1[1]-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, lineType=cv2.LINE_AA)

cv2.imwrite("output_check.jpg", img)
print("Saved: output_check.jpg")
