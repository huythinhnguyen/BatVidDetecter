import sys
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from typing import List, Tuple
from matplotlib.patches import Rectangle


REPO_NAME = 'BatDetecter'
REPO_PATH = os.path.abspath(__file__)
while os.path.basename(REPO_PATH) != REPO_NAME: REPO_PATH = os.path.dirname(REPO_PATH)
if REPO_PATH not in sys.path: sys.path.append(REPO_PATH)

VIDEOS_DIR = os.path.join(REPO_PATH, 'vid')
TEMP_FRAMES_DIR = os.path.join(REPO_PATH, 'temp_frames')

def convert_vid_to_frames(vid_name:str, starting_frame:int=0) -> int: # returns the number of frames
    if not os.path.exists(TEMP_FRAMES_DIR): os.makedirs(TEMP_FRAMES_DIR)
    else: os.system('rm -rf ' + TEMP_FRAMES_DIR + '/*')
    vid_path = os.path.join(VIDEOS_DIR, vid_name)
    cap = cv2.VideoCapture(vid_path)
    frame_count = starting_frame

    while True:
        ret, frame = cap.read()
        if ret:
            name = os.path.join(TEMP_FRAMES_DIR, 'frame{}.jpg'.format(frame_count)) #'./feed1/frame' + str(currentframe) + '.jpg'
            print ('Creating...' + name) 
            cv2.imwrite(name, frame)
            frame_count += 1       
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    return frame_count


def read_frames(read_all:bool = True, starting_frame:int=0, ending_frame:int=0, gray=True) -> List:
    # count number of files inside TEMP_FRAMES_DIR
    num_frames = len([name for name in os.listdir(TEMP_FRAMES_DIR) if os.path.isfile(os.path.join(TEMP_FRAMES_DIR, name))])
    if read_all: ending_frame = num_frames - 1
    current_frame = starting_frame
    frames = []
    while current_frame <= ending_frame:
        if gray:
            frame = cv2.imread(os.path.join(TEMP_FRAMES_DIR, 'frame{}.jpg'.format(current_frame)), cv2.IMREAD_GRAYSCALE)
        else:
            frame = cv2.imread(os.path.join(TEMP_FRAMES_DIR, 'frame{}.jpg'.format(current_frame)))
        frames.append(frame)
        current_frame += 1

    return frames


def compute_frames_derivatives(frames:List) -> List:
    number_of_frames = len(frames)
    fs = np.asarray(frames)
    devs = fs[1:] - fs[:-1]
    if number_of_frames - 1 != len(devs): raise ValueError('Number of frames and derivatives do not match')
    return devs


# create a function that will create and run a matplotlib widget with a slider to view different derivatives
def plot_derivatives(frames:List, derivatives:List) -> None:
    fig, ax = plt.subplots(dpi=150)
    plt.subplots_adjust(left=0.25, bottom=0.25)
    image0 = ax.imshow(frames[0], cmap='gray')
    image1 = ax.imshow(derivatives[0], cmap='jet', alpha=0.5)
    ax.set_title('Frame 0')
    ax.axis('off')

    slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = plt.Slider(slider_ax, 'Frame', 0, len(derivatives)-1, valinit=0, valstep=1)

    def update(val):
        frame = int(slider.val)
        image0.set_data(frames[frame])
        image1.set_data(derivatives[frame])
        ax.set_title('Frame {}'.format(frame))
        ax.axis('off')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def plot_derivatives_with_bb_and_centroids(frames:List, derivatives:List, bounding_boxes:List, centroids:List) -> None:
    fig, ax = plt.subplots(dpi=150)
    plt.subplots_adjust(left=0.25, bottom=0.25)
    image0 = ax.imshow(frames[0], cmap='gray')
    image1 = ax.imshow(derivatives[0], cmap='jet', alpha=0.5)
    ax.set_title('Frame 0')
    ax.axis('off')

    slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = plt.Slider(slider_ax, 'Frame', 0, len(derivatives)-1, valinit=0, valstep=1)

    def update(val):
        frame = int(slider.val)
        image0.set_data(frames[frame])
        image1.set_data(derivatives[frame])
        ax.set_title('Frame {}'.format(frame))
        ax.axis('off')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def plot_derivatives_with_bb_and_centroids(frames: List, derivatives: List, bounding_boxes: List[Tuple[int, int, int, int]]) -> None:
    fig, ax = plt.subplots(dpi=150)
    plt.subplots_adjust(left=0.25, bottom=0.25)
    image0 = ax.imshow(frames[0], cmap='gray')
    image1 = ax.imshow(derivatives[0], cmap='jet', alpha=0.5)
    ax.set_title('Frame 0')
    ax.axis('off')

    slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = plt.Slider(slider_ax, 'Frame', 0, len(derivatives)-1, valinit=0, valstep=1)

    def update(val):
        frame = int(slider.val)
        image0.set_data(frames[frame])
        image1.set_data(derivatives[frame])

        # Clear previous bounding boxes and centroids
        for rect in ax.patches:
            rect.remove()
        # Draw bounding boxes
        #for bb in bounding_boxes[frame]:
        print(bounding_boxes[frame])
        # rect = Rectangle((bounding_boxes[frame][0],
        #                      bounding_boxes[frame][1]),
        #                      bounding_boxes[frame][2],
        #                      bounding_boxes[frame][3],
        #                      linewidth=1, edgecolor='r', facecolor='none')
        
        # ax.add_patch(rect)
        if len(bounding_boxes[frame]) > 0:
            for bb in bounding_boxes[frame]:
                rect = Rectangle((bb[0], bb[1]), bb[2], bb[3], linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
        # new_derv = cv2.rectangle(derivatives[frame], 
        #                          (bounding_boxes[frame][0], bounding_boxes[frame][1]), 
        #                          (bounding_boxes[frame][0] + bounding_boxes[frame][2], bounding_boxes[frame][1] + bounding_boxes[frame][3]), 
        #                          (255, 255, 255), 5)
        # image1.set_data(new_derv)
        # # Draw centroids
        # for centroid in centroids[frame]:
        #     ax.plot(centroid[0], centroid[1], 'go', markersize=5)

        ax.set_title('Frame {}'.format(frame))
        ax.axis('off')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def test1():
    vid_name = 'bat_feed_trim.mp4'
    #num_frames = convert_vid_to_frames(vid_name)
    frames = read_frames()
    derivatives = compute_frames_derivatives(frames)
    # zeros out derivatives values less than 155
    for i in range(20,30): print(np.max(derivatives[i]))
    #derivatives[np.abs(derivatives) < 100] = 0
    # run sand and pepper noise filter on derivatives with a large kernel size
    bounding_boxes_set = []

    for i in range(len(derivatives)):
        derivatives[i] = cv2.medianBlur(derivatives[i], 35)
        derivatives[i][derivatives[i] < 100] = 0
        # Convert the heatmap array to an 8-bit image
        #derivatives[i] = np.uint8(derivatives[i])
        # Threshold the image to identify regions of interest
        _, thresholded = cv2.threshold(derivatives[i], 1, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = []
        # Iterate through contours and extract bounding boxes and centroids
        for contour in contours:
            # Bounding box
            #x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append(cv2.boundingRect(contour))
            # Centroid
            #M = cv2.moments(contour)
            #centroid_x = int(M["m10"] / M["m00"])
            #centroid_y = int(M["m01"] / M["m00"])
            #centroids.append((centroid_x, centroid_y))
            # Print or use the bounding box and centroid as needed
            #print(f"Bounding Box: x={x}, y={y}, width={w}, height={h}")
            #print(f"Centroid: x={centroid_x}, y={centroid_y}")

            # Optionally, draw the bounding box and centroid on the original heatmap image
            #cv2.rectangle(heatmap_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #cv2.circle(heatmap_image, (centroid_x, centroid_y), 2, (0, 0, 255), -1)
        bounding_boxes_set.append(bounding_boxes)




    #plot_derivatives(frames, derivatives)
    plot_derivatives_with_bb_and_centroids(frames, derivatives, bounding_boxes_set)#, centroids)


def main():
    return test1()

if __name__=='__main__':
    main()