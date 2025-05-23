from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # Load a model

video_path = "C:/Users/seenu/OneDrive/Desktop/programming/Python/object_detector/test1.mp4"  # Path to your video file

cap = cv2.VideoCapture(video_path)  # Open the video file

fps = cap.get(cv2.CAP_PROP_FPS)  # Get original FPS
delay = int(1000 / (fps * 1.25))  # Calculate delay for 1.25x speed

ret = True
while ret:
    ret, frame = cap.read()  # Read a frame from the video 
    
    results = model.track(source=frame, persist=True)  # Perform inference on the frame
    
    frame_ =  results[0].plot()  # Get the frame with predictions

    cv2.imshow('frame', frame_)  # Display the frame with predictions
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break
cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows