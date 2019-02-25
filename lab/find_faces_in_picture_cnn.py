import time
import numpy as np
import cv2
import argparse
from pathlib import Path
import face_recognition

# VIDEO_PATH = '../examples/short_hamilton_clip.mp4'
VIDEO_PATH = '/home/ee303/WORKSPACE/FACE/video/lab_cam1_180920_3.avi'
OUTPUT_DIR = 'lab_cam_detection_results/lab_cam1_180920_3'


if __name__ == "__main__":

    # Create output dir if not it's not exist
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # Read video file
    video_capture = cv2.VideoCapture(VIDEO_PATH)

    # Get dimension, frame rate and frame count of the video
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH ))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT ))
    fps =  video_capture.get(cv2.CAP_PROP_FPS)
    length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_movie = cv2.VideoWriter(f'{OUTPUT_DIR}/output.avi', fourcc, fps, (width, height))

    # Frame counter
    frame_count = 0

    with open(f'{OUTPUT_DIR}/face_locations.csv', 'a+', encoding='utf-8') as f:

        # Start time
        start = time.time()

        while video_capture.isOpened():

            # Grab a single frame of video
            ret, frame = video_capture.read()

            # Bail out when the video file ends
            if not ret:
                break

            # Aggregate frame count
            frame_count += 1

            # Use GPU to detect faces in RGB frame
            face_locations = face_recognition.face_locations(frame[:, :, ::-1], number_of_times_to_upsample=0, model="cnn")

            for num, face_location in enumerate(face_locations):

                # Print the location of each face in this image
                top, right, bottom, left = face_location
                print(f"Frame: {frame_count}  --  Top: {top}, Left: {left}, Bottom: {bottom}, Right: {right}")

                # Print location information to recording file
                print(f"{frame_count}, {top}, {left}, {bottom}, {right}", file=f)

                # Draw detection result on frame and save cropped face image
                face_image = frame[top:bottom, left:right]
                cv2.imwrite(f'{OUTPUT_DIR}/face_result_{frame_count}_{num}.jpg', face_image)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Show video with detection result
            cv2.imshow('result', frame)
            cv2.waitKey(1)

            # Save video with detection result
            output_movie.write(frame)

        # End time
        end = time.time()

        # Calculate frames per second
        seconds = end - start
        frame_rate  = length / seconds
        print(f"Estimated time elapsed: {seconds:.2f} -- Frame rate: {frame_rate:.2f}")
