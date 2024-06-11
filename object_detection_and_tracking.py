import cv2
import numpy as np

class ShapeDetector:
    # Constructor
    def __init__(self, hough_params, threshold_value, max_value, threshold_type):
        self.hough_params = hough_params
        self.threshold_value = threshold_value
        self.max_value = max_value
        self.threshold_type = threshold_type

    def detect_shapes(self, image):
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        circles = self.detect_circles(blurred)
        squares = self.detect_squares(blurred)

        return circles + squares

    # Method to detect circles
    def detect_circles(self, blurred):
        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, **self.hough_params)
        detected_circles = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                detected_circles.append((x - r, y - r, 2 * r, 2 * r, "Circle", (x, y)))
        return detected_circles

    # Method to detect squares
    def detect_squares(self, blurred):
        _, thresh = cv2.threshold(blurred, self.threshold_value, self.max_value, self.threshold_type)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_squares = []
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                detected_squares.append((x, y, w, h, "Square", (x + w // 2, y + h // 2)))
        return detected_squares


class Tracker:
    # Constructor
    def __init__(self, video_path, hough_params, threshold_value, max_value, threshold_type):
        self.video_path = video_path
        self.shape_detector = ShapeDetector(hough_params, threshold_value, max_value, threshold_type)

    def detect_shapes_in_video(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        path = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Detect shapes in the frame
            shapes = self.shape_detector.detect_shapes(frame)

            # Print coordinates of detected shapes in each frame
            print(f"Frame {frame_count}:")
            for shape in shapes:
                print(shape)
                # Append the coordinates of the detected shape to the path
                path.append(shape[-1])

            # Display frame with bounding boxes around detected shapes
            for shape in shapes:
                x, y, w, h, shape_type, _ = shape
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, shape_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Visualize the tracking path
            for i in range(1, len(path)):
                cv2.circle(frame, path[i], 2, (0, 0, 255), -1)

            cv2.imshow('Object Detection and Tracking (press q to exit)', frame)

            # Stop if pressed 'q'
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Release video capture and close all windows
        cap.release()
        cv2.destroyAllWindows()


# Main method
if __name__ == "__main__":
    hough_params = {
        'dp': 1,
        'minDist': 20,
        'param1': 50,
        'param2': 30,
        'minRadius': 10,
        'maxRadius': 100
    }
    threshold_value = 40
    max_value = 255
    threshold_type = cv2.THRESH_BINARY
    video_detector = Tracker("./luxonis_task_video.mp4", hough_params, threshold_value, max_value, threshold_type)
    video_detector.detect_shapes_in_video()
