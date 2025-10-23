import cv2
import numpy as np

def read_events(file_path):
    events = []
    with open(file_path, 'r') as file:
        for line in file:
            t, x, y, p = map(float, line.split())
            if 1 < t < 2:
                p = 1 if p == 1 else -1
                events.append((t, int(x), int(y), p))
    return events

def event_frame(coords, polarities, image_shape):
    image = np.ones(image_shape) * 127
    image = image.astype(np.uint8)
    for (x, y), p in zip(coords, polarities):
        if p == 1:
            image[y, x] = 255
        else:
            image[y, x] = 0
    return image

def generate_event_frames(events, tau, image_shape):
    temp_coords = []
    temp_polarities = []
    start_time = events[0][0]

    for t, x, y, p in events:
        temp_coords.append((x, y))
        temp_polarities.append(p)

        if t - start_time >= tau:
            img = event_frame(temp_coords, temp_polarities, image_shape)
            cv2.imshow('Event Frame', img)
            cv2.waitKey(0)  # Wait for key press to show next frame
            temp_coords.clear()
            temp_polarities.clear()
            start_time = t

events_file = r'c:\Users\anton\Documents\Erasmus\AVS\Lab13\events.txt'
events = read_events(events_file)
image_shape = (180, 240)  
tau = 0.01 

generate_event_frames(events, tau, image_shape)

# tau = 0.001 the frames show finer temporal resolution capturing more granular changes but appear noisy
# tau = 0.01 the frames balance temporal resolution and smoothness providing clear object movement without excessive noise
# tau = 0.1 the frames aggregate more events showing smoother object movement but with potential loss of fine temporal details
