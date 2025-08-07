import cv2
import cv2.aruco as aruco

capture = cv2.VideoCapture(0)

circle_tracker = []

def findAruco(img, marker_size=4, total_markers=250):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{marker_size}X{marker_size}_{total_markers}')
    arucoDict = aruco.getPredefinedDictionary(key)
    
    # Compatível com versões mais recentes do OpenCV
    try:
        # Para OpenCV 4.7+
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(arucoDict, parameters)
        bbox, ids, _ = detector.detectMarkers(gray)
    except AttributeError:
        # Para versões mais antigas do OpenCV
        parameters = aruco.DetectorParameters_create()
        bbox, ids, _ = aruco.detectMarkers(gray, arucoDict, parameters=parameters)
    
    if ids is not None:
        for i in range(len(ids)):
            id = ids[i][0]
            bbox_points = bbox[i][0]

            # Center of the marker
            cx = int((bbox_points[0][0] + bbox_points[1][0] + bbox_points[2][0] + bbox_points[3][0]) / 4)
            cy = int((bbox_points[0][1] + bbox_points[1][1] + bbox_points[2][1] + bbox_points[3][1]) / 4)

            # Update the position of the circle to the center of the marker
            circle_tracker.append((cx, cy))

    return bbox, ids


while True:
    ret, img = capture.read()
    if not ret:
        print("Failed to capture image")
        break

    bbox, ids = findAruco(img)
    
    # Draw circles based on marker positions
    for coord in circle_tracker:
        cv2.circle(img, coord, 5, (0, 0, 255), -1)  

    cv2.imshow("img", img)

    key = cv2.waitKey(1)
    if key == 27:  # Escape key
        break
    elif key & 0xFF == ord('q'):
        circle_tracker = []

# Release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()