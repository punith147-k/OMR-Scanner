import cv2
import numpy as np


# Function to reorder corner points in a specific order
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))  # Reshape points into a 4x2 matrix
    myPointsNew = np.zeros((4, 1, 2), np.int32)  # Create a placeholder for reordered points
    add = myPoints.sum(1)  # Calculate the sum of x and y coordinates
    myPointsNew[0] = myPoints[np.argmin(add)]  # Top-left corner
    myPointsNew[3] = myPoints[np.argmax(add)]  # Bottom-right corner
    diff = np.diff(myPoints, axis=1)  # Calculate the difference between x and y coordinates
    myPointsNew[1] = myPoints[np.argmin(diff)]  # Top-right corner
    myPointsNew[2] = myPoints[np.argmax(diff)]  # Bottom-left corner
    return myPointsNew


# Function to filter rectangular contours
def rectContour(contours):
    rectCon = []
    for i in contours:
        area = cv2.contourArea(i)  # Calculate contour area
        if area > 50:  # Ignore small contours
            peri = cv2.arcLength(i, True)  # Calculate contour perimeter
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)  # Approximate polygonal curves
            if len(approx) == 4:  # Keep only rectangular contours
                rectCon.append(i)
    rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True)  # Sort contours by area
    return rectCon


# Function to get the corner points of a contour
def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True)  # Calculate perimeter
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True)  # Approximate polygonal curves
    return approx


# Function to split an image into boxes for OMR processing
def splitBoxes(img):
    rows = np.vsplit(img, 5)  # Split the image into 5 rows
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 5)  # Split each row into 5 columns
        for box in cols:
            boxes.append(box)  # Collect all boxes
    return boxes


# Function to draw a grid over an image
def drawGrid(img, questions=5, choices=5):
    secW = int(img.shape[1] / questions)  # Section width
    secH = int(img.shape[0] / choices)  # Section height
    for i in range(0, 9):
        pt1 = (0, secH * i)
        pt2 = (img.shape[1], secH * i)
        pt3 = (secW * i, 0)
        pt4 = (secW * i, img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0), 2)  # Draw horizontal lines
        cv2.line(img, pt3, pt4, (255, 255, 0), 2)  # Draw vertical lines
    return img


# Function to display answers on the processed image
def showAnswers(img, myIndex, grading, ans, questions=5, choices=5):
    secW = int(img.shape[1] / questions)  # Section width
    secH = int(img.shape[0] / choices)  # Section height
    for x in range(0, questions):
        myAns = myIndex[x]  # User's selected answer
        cX = (myAns * secW) + secW // 2
        cY = (x * secH) + secH // 2
        if grading[x] == 1:  # Correct answer
            myColor = (0, 255, 0)  # Green color
            cv2.circle(img, (cX, cY), 50, myColor, cv2.FILLED)
        else:  # Incorrect answer
            myColor = (0, 0, 255)  # Red color
            cv2.circle(img, (cX, cY), 50, myColor, cv2.FILLED)
            myColor = (0, 255, 0)
            correctAns = ans[x]  # Display correct answer
            cv2.circle(img, ((correctAns * secW) + secW // 2, (x * secH) + secH // 2),
                       20, myColor, cv2.FILLED)


# Function to create a side-by-side display of two images
def create_side_by_side_display(img1, img2):
    h1, w1 = img1.shape[:2]  # Get dimensions of first image
    h2, w2 = img2.shape[:2]  # Get dimensions of second image
    total_width = w1 + w2
    max_height = max(h1, h2)
    display = np.zeros((max_height, total_width, 3), dtype=np.uint8)  # Create a blank canvas
    display[:h1, :w1] = img1  # Place the first image
    display[:h2, w1:w1 + w2] = img2  # Place the second image
    return display


def main():
    webCamFeed = True  # Flag to use webcam feed
    pathImage = "5.jpg"  # Path to image if webcam not used
    cap = cv2.VideoCapture(0)
    cap.set(10, 160)  # Set webcam brightness
    heightImg = 700  # Height of the processed image
    widthImg = 700  # Width of the processed image
    questions = 5
    choices = 5
    ans = [1, 2, 0, 2, 4]  # Correct answers

    count = 0

    while True:
        if webCamFeed:
            success, img = cap.read()  # Capture frame from webcam
        else:
            img = cv2.imread(pathImage)  # Read image from file

        img = cv2.resize(img, (widthImg, heightImg))  # Resize image
        imgFinal = img.copy()  # Copy for final display

        # Preprocess the image
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
        imgCanny = cv2.Canny(imgBlur, 10, 70)

        try:
            # Find contours and process them
            contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            rectCon = rectContour(contours)  # Filter rectangular contours
            biggestPoints = getCornerPoints(rectCon[0])  # Get corner points of the biggest rectangle
            gradePoints = getCornerPoints(rectCon[1])  # Get corner points of the grading area

            if biggestPoints.size != 0 and gradePoints.size != 0:
                # Warp perspective for processing
                biggestPoints = reorder(biggestPoints)
                pts1 = np.float32(biggestPoints)
                pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
                matrix = cv2.getPerspectiveTransform(pts1, pts2)
                imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

                # Process and grade the answers
                imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
                imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]
                boxes = splitBoxes(imgThresh)

                myPixelVal = np.zeros((questions, choices))
                countR = 0
                countC = 0
                for image in boxes:
                    totalPixels = cv2.countNonZero(image)  # Count non-zero pixels in the box
                    myPixelVal[countR][countC] = totalPixels
                    countC += 1
                    if countC == choices:
                        countC = 0
                        countR += 1

                myIndex = []
                for x in range(0, questions):
                    arr = myPixelVal[x]
                    myIndexVal = np.where(arr == np.amax(arr))  # Find the darkest box (marked answer)
                    myIndex.append(myIndexVal[0][0])

                grading = []
                for x in range(0, questions):
                    if ans[x] == myIndex[x]:  # Compare user's answer with correct answer
                        grading.append(1)
                    else:
                        grading.append(0)
                score = (sum(grading) / questions) * 100  # Calculate score

                # Display results
                showAnswers(imgWarpColored, myIndex, grading, ans)
                drawGrid(imgWarpColored)
                imgRawDrawings = np.zeros_like(imgWarpColored)
                showAnswers(imgRawDrawings, myIndex, grading, ans)
                invMatrix = cv2.getPerspectiveTransform(pts2, pts1)
                imgInvWarp = cv2.warpPerspective(imgRawDrawings, invMatrix, (widthImg, heightImg))

                imgRawGrade = np.zeros_like(img, np.uint8)
                cv2.putText(imgRawGrade, str(int(score)) + "%", (70, 100),
                            cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 255), 3)

                imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)

                # Show results side by side
                display = create_side_by_side_display(img, imgFinal)
                cv2.imshow("OMR Scanner", display)

        except:
            # If an error occurs, just show the original image
            display = create_side_by_side_display(img, img)
            cv2.imshow("OMR Scanner", display)

        # Save the result when 's' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite("Scanned/myImage" + str(count) + ".jpg", imgFinal)
            count += 1

        # Quit the application when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    if webCamFeed:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
