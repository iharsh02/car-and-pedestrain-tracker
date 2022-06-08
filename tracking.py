#  this is cars and pedestrian tracking

import cv2

# our image

# img_file = 'trafic.jpgresources\trafic.jpg'
# ou video file

video = cv2.VideoCapture(0) # this will detect the your live camera footage 

# our pre trained classifier

car_tracker_file = 'tracker\car_detector.xml'
pedestrain_tracker_file = 'tracker\pedestrain_tracker.xml'

# create car classifier

car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrain_tracker = cv2.CascadeClassifier(pedestrain_tracker_file)


#  now we need to add a loop that will run till we manualy stops it or any kind of the program error

while True:

    # this will read the current frame of the video

    (read_successful, frame) = video.read()

    if read_successful:
        # this will convert it to the grayscale

        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    else:
        break

    # cv2.resizeWindow(frame,640,240)
#  detect car and pedestrian

    cars = car_tracker.detectMultiScale(grayscale_frame)
    pedestrain = pedestrain_tracker.detectMultiScale(grayscale_frame)
    print(pedestrain)


#  now draw rectange arround the cars

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
    
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

#  draw rectangle around the pedestrian
    for (x, y, w, h) in pedestrain:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)

#  display the image with the faces spotted
    cv2.imshow('car and pedestian tracker ', frame)

#  dont auto close the code

    key =  cv2.waitKey(1)

    # stoped if q is pressed

    if key == 81 or key == 113: 
        break

#  release video capture 
video.release()
print('code completed')