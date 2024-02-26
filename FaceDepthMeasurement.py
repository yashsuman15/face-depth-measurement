import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)

while cap.isOpened():
    success, img = cap.read()
    img = cv2.flip(img, flipCode=1)
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        # step-1->> find the eyes point on the screen----------------------------
        face = faces[0]  # we take the 1st face detected
        right_Eye_point = face[145]
        left_Eye_point = face[374]

        # step-2->> draw the eyes point and distance between them------------------------

        # cv2.line(img, right_Eye_point, left_Eye_point, (0, 150, 0), 2)
        # cv2.circle(img, right_Eye_point, 5, (255, 0, 255), cv2.FILLED)
        # cv2.circle(img, left_Eye_point, 5, (255, 0, 255), cv2.FILLED)

        # step-3->> find the distance between eyes point on the screen(in pixels)-------------------------

        w, _ = detector.findDistance(right_Eye_point, left_Eye_point)  # Distance between the points Image
        W = 6.3  # distance between eyes in real world (in cm)

        # step-4->> finding the focal length(f) of camera--------------------------------

        # d = 30  # distance between camera and face
        # f = (w*d)/W  # focal length = 600 at d = 30

        # step-5->> finding distance(d) using the focal length--------------------------------
        f = 600
        d = (W * f) / w
        # print(d)

        # step-6->> showing distance on screen
        cv2.putText(img, f"Face Distance = {int(d)}cm", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, 1)
        cvzone.putTextRect(img, f"Face Distance:{int(d)}cm", (face[10][0] - 100, face[10][1] - 30), 1, 1)

    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
