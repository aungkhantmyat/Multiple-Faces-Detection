import mediapipe as mp
import cv2
import time as time
import math
import random
import os
import json
import shutil

start_time = 0
end_time = 0
prev_state = "Only one person is detected"
flag = False

mpFaceDetection = mp.solutions.face_detection  # Detect the face
mpDraw = mp.solutions.drawing_utils  # Draw the required Things for BBox
faceDetection = mpFaceDetection.FaceDetection(0.75)
# It has 0 to 1 (Change this to make it more detectable) Default is 0.5 and higher means more detection.
cap = cv2.VideoCapture(0)
video = str(random.randint(1, 50000)) + "MTOPViolation.avi"
writer =cv2.VideoWriter(video , cv2.VideoWriter_fourcc(*"XVID"), 20, (640, 480))

def MTOP_record_duration(text, img):
    global start_time, end_time, recorded_durations, prev_state, flag, writer, video
    if text != 'Only one person is detected' and prev_state == 'Only one person is detected':
        start_time = time.time()
        writer.write(img)
    elif text != 'Only one person is detected' and str(text) == prev_state and (time.time() - start_time) > 3:
        flag = True
        writer.write(img)
    elif text != 'Only one person is detected' and str(text) == prev_state and (time.time() - start_time) <= 3:
        flag = False
        writer.write(img)
    else:
        if prev_state != "Only one person is detected":
            writer.release()
            end_time = time.time()
            duration = math.ceil(end_time - start_time)
            HeadViolation = {
                "Name": prev_state,
                "Time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
                "Duration": str(duration) + " seconds",
                "Mark": (2 * (duration - 3)),
                "Link": video
            }
            if flag:
                write_json(HeadViolation)
                move_file_to_output_videos(video)
            else:
                os.remove(video)
            video = str(random.randint(1, 50000)) + "MTOPViolation.avi"
            writer =cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"XVID"), 20, (640, 480))
            flag = False
    prev_state = text

# function to add to JSON
def write_json(new_data, filename='violation.json'):
    with open(filename,'r+') as file:
          # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data.append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent = 4)

def move_file_to_output_videos(file_name):
    # Get the current working directory (project folder)
    current_directory = os.getcwd()
    # Define the paths for the source file and destination folder
    source_path = os.path.join(current_directory, file_name)
    destination_path = os.path.join(current_directory, 'OutputVideos', file_name)
    try:
        # Use 'shutil.move' to move the file to the destination folder
        shutil.move(source_path, destination_path)
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found in the project folder.")
    except shutil.Error as e:
        print(f"Error: Failed to move the file. {e}")

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    if results.detections:
        for id, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                int(bboxC.width * iw), int(bboxC.height * ih)
            # its not dividing, Getting to next line

            # Drawing the recantangle
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            #cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 10)

        if id > 0:
            text = "More than one person is detected."
        else:
            text = "Only one person is detected"

    MTOP_record_duration(text,img)
    cv2.imshow("Video", img)
    cv2.waitKey(1)
    # Check if 'finish' key is pressed (q key) or Esc key is clicked
    key = cv2.waitKey(1)
    if key == ord("q") or key == 27 :
        break
cv2.destroyAllWindows()
cap.release()
