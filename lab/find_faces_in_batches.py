import face_recognition
import cv2

# This code finds all faces in a list of images using the CNN model.
#
# This demo is for the _special case_ when you need to find faces in LOTS of images very quickly and all the images
# are the exact same size. This is common in video processing applications where you have lots of video frames
# to process.
#
# If you are processing a lot of images and using a GPU with CUDA, batch processing can be ~3x faster then processing
# single images at a time. But if you aren't using a GPU, then batch processing isn't going to be very helpful.
#
# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read the video file.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Open video file
# video_capture = cv2.VideoCapture("short_hamilton_clip.mp4")
video_capture = cv2.VideoCapture("../../video/lab_cam1_180920_1.avi")
# video_capture = cv2.VideoCapture("http://avlabadmin:avlab32343738@140.118.58.27/img/video.mjpeg")
# video_capture = cv2.VideoCapture("http://140.118.132.21/img/video.mjpeg?channel=0")

frames = []
frame_count = 0

while video_capture.isOpened():
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Bail out when the video file ends
    if not ret:
        break

    frame = cv2.resize(frame, (480, 270))
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    frame = frame[:, :, ::-1]

    # Save each frame of the video to a list
    frame_count += 1
    frames.append(frame)

    # Every 128 frames (the default batch size), batch process the list of frames to find faces
    if len(frames) == 1:
        batch_of_face_locations = face_recognition.batch_face_locations(frames, number_of_times_to_upsample=0)

        # Now let's list all the faces we found in all 128 frames
        for frame_number_in_batch, face_locations in enumerate(batch_of_face_locations):
            number_of_faces_in_frame = len(face_locations)

            frame_number = frame_count - 128 + frame_number_in_batch
            print("I found {} face(s) in frame #{}.".format(number_of_faces_in_frame, frame_number))

            for face_location in face_locations:
                # Print the location of each face in this frame
                top, right, bottom, left = face_location
                print(" - A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

                cv2.rectangle(frames[frame_number_in_batch][:, :, ::-1], (left, top), (right, bottom), (0, 0, 255), 2)
                face = frames[frame_number_in_batch][top:bottom, left:right, ::-1]
                # cv2.imwrite(f'face_result_{frame_number}.jpg', face)
            cv2.imshow('frame', frames[frame_number_in_batch][:, :, ::-1])
            cv2.waitKey(1)

        # Clear the frames array to start the next batch
        frames = []
