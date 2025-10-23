import numpy as np
import cv2
import imutils
import os
from os.path import join

DATASET_DIR = r'c:\Users\anton\Documents\Erasmus\AVS\Lab7\sequences'

SIGMA = 17
SEARCH_REGION_SCALE = 2
LR = 0.125
NUM_PRETRAIN = 128
VISUALIZE = True

def load_gt(gt_file):

    with open(gt_file, 'r') as file:
        lines = file.readlines()

    lines = [line.split(',') for line in lines]
    lines = [[int(float(coord)) for coord in line] for line in lines]
    # returns in x1y1wh format
    return lines

def crop_search_window(bbox, frame):
    xmin, ymin, width, height = bbox
    xmax = xmin + width
    ymax = ymin + height

    # TODO (1): Modify xmin, xmax, ymin, ymax to include a wider image context
    search_scale = SEARCH_REGION_SCALE
    xmin -= width * (search_scale - 1) / 2
    xmax += width * (search_scale - 1) / 2
    ymin -= height * (search_scale - 1) / 2
    ymax += height * (search_scale - 1) / 2

    # TODO (2): Protect against extending search area beyond frame by padding
    padding = int(max(width, height) * (search_scale - 1) / 2)
    frame = cv2.copyMakeBorder(frame, padding, padding, padding, padding, cv2.BORDER_REFLECT)
    xmin += padding
    xmax += padding
    ymin += padding
    ymax += padding

    # Ensure coordinates are within frame boundaries
    xmin = max(0, xmin)
    xmax = min(frame.shape[1], xmax)
    ymin = max(0, ymin)
    ymax = min(frame.shape[0], ymax)

    # Crop the search window
    window = frame[int(ymin) : int(ymax), int(xmin) : int(xmax), :]
    window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)

    return window

def get_gauss_response(gt_box):

    width = gt_box[2] * SEARCH_REGION_SCALE
    height = gt_box[3] * SEARCH_REGION_SCALE
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))

    center_x = width // 2
    center_y = height // 2
    dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * SIGMA)
    response = np.exp(-dist)

    return response

def pre_process(img):
    height, width = img.shape
    img = img.astype(np.float32)

    # TODO (3): Apply logarithmic transformation
    img = np.log(img + 1)

    # TODO (3): Normalize the image
    mean_val = np.mean(img)
    std_val = np.std(img)
    img = (img - mean_val) / std_val

    # Apply 2D Hanning window
    win_col = np.hanning(width)
    win_row = np.hanning(height)
    mask_col, mask_row = np.meshgrid(win_col, win_row)
    window = mask_col * mask_row
    img = img * window

    return img

def random_warp(img):
    # Generate a random angle in the range -15 to +15 degrees
    angle = np.random.uniform(-15, 15)

    # Rotate the image by the drawn angle
    img_rot = imutils.rotate_bound(img, angle)

    # Scale the rotated image to the original size of the input image
    img_resized = cv2.resize(img_rot, (img.shape[1], img.shape[0]))

    # Visualize the result
    cv2.imshow("Random Warp", img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img_resized

# Test the random_warp function visually
img = cv2.imread(r'c:\Users\anton\Documents\Erasmus\AVS\Lab7\sequences\jump\color\00000001.jpg', cv2.IMREAD_GRAYSCALE)  # Load a grayscale image
warped_img = random_warp(img)  # Apply random warp transformation

def initialize(init_frame, init_gt):

    g = get_gauss_response(init_gt)
    G = np.fft.fft2(g)
    Ai, Bi = pre_training(init_gt, init_frame, G)

    return Ai, Bi, G

def pre_training(init_gt, init_frame, G):

    template = crop_search_window(init_gt, init_frame)
    fi = pre_process(template)
    
    Ai = G * np.conjugate(np.fft.fft2(fi))                # (1a)
    Bi = np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))  # (1b)

    for _ in range(NUM_PRETRAIN):
        fi = pre_process(random_warp(template))

        Ai = Ai + G * np.conjugate(np.fft.fft2(fi))               # (1a)
        Bi = Bi + np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi)) # (1b)

    return Ai, Bi

def track(image, position, Ai, Bi, G):

    response = predict(image, position, Ai/Bi)
    new_position = update_position(response, position)
    newAi, newBi = update(image, new_position, Ai, Bi, G)

    return new_position, newAi, newBi

def predict(frame, position, H):
    # Extract the search window
    search_window = crop_search_window(position, frame)
    # Preprocess the search window
    fi = pre_process(search_window)
    # Compute the filter response
    response = np.fft.ifft2(np.fft.fft2(fi) * H)
    gi = np.real(response)  # Take the real part of the inverse Fourier transform
    return gi

def update(frame, position, Ai, Bi, G):
    # Extract the search window
    search_window = crop_search_window(position, frame)
    # Preprocess the search window
    fi = pre_process(search_window)
    # Compute the correlation response
    response = np.fft.ifft2(np.fft.fft2(fi) * np.conjugate(Ai) / Bi)
    # Compute the updated filter parameters
    newAi = LR * G * np.conjugate(np.fft.fft2(fi)) + (1 - LR) * Ai
    newBi = LR * np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi)) + (1 - LR) * Bi
    return newAi, newBi

def update_position(spatial_response, position):
    max_val = np.max(spatial_response)
    max_idx = np.where(spatial_response == max_val)
    avg_idx = np.mean(max_idx, axis=1)
    # Compute the shift from the center of the filter response
    shift_x, shift_y = avg_idx - np.array(spatial_response.shape) / 2
    new_x, new_y, width, height = position
    # Update the position coordinates
    new_x += shift_x
    new_y += shift_y
    new_position = (int(new_x), int(new_y), width, height)
    return new_position

def bbox_iou(box1, box2):
    # Transform from center and width to exact coordinates
    b1_x1, b1_x2 = box1[0], box1[0] + box1[2]
    b1_y1, b1_y2 = box1[1], box1[1] + box1[3]
    b2_x1, b2_x2 = box2[0], box2[0] + box2[2]
    b2_y1, b2_y2 = box2[1], box2[1] + box2[3]

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)
    # Intersection area
    inter_area = np.clip(inter_rect_x2 - inter_rect_x1, a_min=0, a_max=None) * np.clip(inter_rect_y2 - inter_rect_y1, a_min=0, a_max=None)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def test_sequence(DATASET_DIR, sequence):

    seqdir = join(DATASET_DIR, sequence)
    imgdir = join(seqdir, 'color')
    imgnames = os.listdir(imgdir)                  
    imgnames.sort()

    print('init frame:', join(imgdir, imgnames[0]))
    init_img = cv2.imread(join(imgdir, imgnames[0]))
    gt_boxes = load_gt(join(seqdir, 'groundtruth.txt'))
    position = gt_boxes[0]
    Ai, Bi, G = initialize(init_img, position)

    if VISUALIZE:
        cv2.rectangle(init_img, (position[0], position[1]), (position[0]+position[2], position[1]+position[3]), (255, 0, 0), 2)
        cv2.imshow('demo', init_img)
        cv2.waitKey(0)

    results = []
    total_iou = 0  # Initialize total_iou
    for idx, imgname in enumerate(imgnames[1:], start=1):
        img = cv2.imread(join(imgdir, imgname))
        position, Ai, Bi = track(img, position, Ai, Bi, G)
        results.append(list(position))  # Convert position to a list before appending

        iou = bbox_iou(position, gt_boxes[idx])
        total_iou += iou

        if VISUALIZE:
            position = [round(x) for x in position]
            cv2.rectangle(img, (position[0], position[1]), (position[0] + position[2], position[1] + position[3]), (255, 0, 0), 2)
            cv2.imshow('demo', img)
            if cv2.waitKey(0) == ord('q'):
                break

    average_iou = total_iou / len(imgnames[1:])
    return results, gt_boxes, average_iou

sequences = ['sunshade']
ious_per_sequence = {}
for sequence in sequences:

    results, gt_boxes, average_iou = test_sequence(DATASET_DIR, sequence)
    ious_per_sequence[sequence] = average_iou
    print(sequence, ':', average_iou)

print('Mean IoU:', np.mean(list(ious_per_sequence.values())))
