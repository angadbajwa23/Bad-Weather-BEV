import cv2
import numpy as np
import torch
import math
import os
import warnings
import ssl
import urllib.request
import time

warnings.filterwarnings("ignore", category=UserWarning)

# Disable SSL certificate verification
ssl_context = ssl._create_unverified_context()

# Set SSL certificate file path
os.environ['SSL_CERT_FILE'] = 'cacert.pem'

# Load the YOLOv5 model
model_path = 'yolov5s.pt'
device = 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model.to(device)

# Load the video
video = cv2.VideoCapture('derained.mp4')
#output_filename = 'output_yolo.mp4'
#width, height = 1280, 720
#videoOut = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
output_filename = 'derained_obj.mp4'
width, height = 1280, 720
videoOut = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width * 2, height))  # Updated video dimensions



track = True
#track = False



yolo_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]


def overlay_transparent(background, foreground, angle, x, y, objSize=50):
    original_frame = background.copy()
    foreground = cv2.resize(foreground, (objSize, objSize))

    # Get the shape of the foreground image
    rows, cols, channels = foreground.shape
    #print("Foreground Shape:", rows, cols, channels)

    #if channels < 4:
     #   print("Foreground image does not have an alpha channel")
      #  return background
    alpha_channel = foreground[:, :, 3] / 255.0 

    # Calculate the center of the foreground image
    center_x = int(cols / 2)
    center_y = int(rows / 2)
    #print("Center Coordinates:", center_x, center_y)

   

    # Rotate the foreground image
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
    #print(M)
    foreground = cv2.warpAffine(foreground, M, (cols, rows))

    # Overlay the rotated foreground image onto the background image
    for row in range(rows):
        for col in range(cols):
            if x + row < background.shape[0] and y + col < background.shape[1]:
                alpha = alpha_channel[row, col]
                if alpha > 0:  # Only blend if the alpha value is greater than 0
                    foreground_color = foreground[row, col, :3]
                    background_color = background[x + row, y + col]
                    blended_color = (alpha * foreground_color) + ((1 - alpha) * background_color)
                    #print("Alpha:", alpha)
                    #print("Foreground Color:", foreground_color)
                    #print("Background Color:", background_color)
                    #print("Blended Color:", blended_color)
                    background[x + row, y + col] = blended_color.astype(np.uint8)
                #alpha = foreground[row, col, 3] / 255.0
                #print("Alpha Value:", alpha)
                background[x + row, y + col] = alpha * foreground[row, col, :3] + (1 - alpha) * background[x + row, y + col]
                #background[x + row, y + col] =  foreground[row, col, :3] +  background[x + row, y + col]

    # Blend the foreground and background ROI using cv2.addWeighted
    result = background

    return result


def simulate_object(background, object_class, x, y):

    object_img = cv2.imread(f'assets/box.png', cv2.IMREAD_UNCHANGED)
    
    # Print dimensions of the object image
    #print(f"Object Image Dimensions: {object_img.shape}")
    
    # Simulate the object by overlaying it onto the background image
    overlay_result = overlay_transparent(background[y:y+object_img.shape[0], x:x+object_img.shape[1]], object_img, 0, 0, 0)
    background[y:y+object_img.shape[0], x:x+object_img.shape[1]] = overlay_result
    
    # Print dimensions of the resulting image
    #print(f"Resulting Image Dimensions: {background.shape}")
    
    # Save resulting image for inspection
    cv2.imwrite("resulting_image.png", background)

    # Load the object image based on the class
    #object_img = cv2.imread(f'assets/{object_class}.png', cv2.IMREAD_UNCHANGED)
    #if object_img is None:
     #   print(f"Failed to load image for class: {object_class}")
     #   return background
    
    #print(f"Loaded image for class: {object_class}, Size: {object_img.shape}, Position: ({x}, {y})")
    
    # Simulate the object by overlaying it onto the background image
    #background[y:y+object_img.shape[0], x:x+object_img.shape[1]] = overlay_transparent(background[y:y+object_img.shape[0], x:x+object_img.shape[1]], object_img, 0, 0, 0)

    return background



def add_myCar_overlay(background):
    overlay_img = cv2.imread('assets/MyCar.png', cv2.IMREAD_UNCHANGED)
    # Get the shape of the overlay image
    rows, cols, _ = overlay_img.shape
    x = 550
    y = background.shape[0] - 200

    # Overlay the image onto the background
    overlay_img = overlay_transparent(background[y:y+rows, x:x+cols], overlay_img, 0, 0, 0, objSize=250)
    background[y:y+rows, x:x+cols] = overlay_img

    return background


def plot_object_bev(transformed_image_with_centroids, src_points ,dst_points , objs_):
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    persObjs = []

    ## mark objs and ids
    for obj_ in objs_:
        #if obj_:
        # Create a numpy array of the centroid coordinates
        centroid_coords = np.array([list(obj_[0])], dtype=np.float32)

            # Apply the perspective transformation to the centroid coordinates
        transformed_coords = cv2.perspectiveTransform(centroid_coords.reshape(-1, 1, 2), M)
        transformed_coords_ = tuple(transformed_coords[0][0].astype(int))

        persObjs.append([transformed_coords_, obj_[1]])
        #print(persObjs)

    return transformed_image_with_centroids, persObjs


frame_count = 0
centroid_prev_frame = []
tracking_objects = {}
tracking_id = 0


# Process each frame of the video
while True:
    start_time = time.time()
    # Read the next frame
    success, frame = video.read()
    frame = cv2.resize(frame, (width, height))
    frame_count += 1
    if not success:
        break

    # Perform object detection on the frame
    results = model(frame, size=320)
    detections = results.pred[0]
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference Time: {inference_time} seconds")

    #print(detections)
    # Create a black image with the same size as the video frames
    image_ = np.zeros((height, width, 3), dtype=np.uint8)
    simulated_image = image_.copy()
    transformed_image_with_centroids = image_.copy()
    transformed_image_to_sim = image_.copy()
    simObjs = image_.copy()

    objs = []
    centroid_curr_frame = []

    #####################
    ##  OBJ DETECTION  ##
    #####################
    for detection in detections:    
        xmin    = detection[0]
        ymin    = detection[1]
        xmax    = detection[2]
        ymax    = detection[3]
        score   = detection[4]
        class_id= detection[5]
        #print(class_id)
        centroid_x = int(xmin + xmax) // 2
        centroid_y =  int(ymin + ymax) // 2

        if int(class_id) in [0, 1, 2, 3, 5, 7] and score >= 0.3:
            # Draw bounding box on the frame
            color = (0, 0, 255)
            object_label = f"{class_id}: {score:.2f}"
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
            cv2.putText(frame, object_label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
            centroid_curr_frame.append([(centroid_x, centroid_y), yolo_classes[int(class_id)]])
            #if track:
             #   objs.append([(centroid_x, centroid_y), yolo_classes[int(class_id)]])
            #if track:
            objs.append([(centroid_x, centroid_y), yolo_classes[int(class_id)]])
            #print("objs")
            #print(objs)


    #####################
    ## OBJECT TRACKING ##
    #####################
    if track:
        if frame_count <= 2:
            for pt1, class_id in centroid_curr_frame:
                for pt2, class_id in centroid_prev_frame:
                    dist = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
                    if dist < 50:
                        tracking_objects[tracking_id] = pt1, class_id
                        tracking_id += 1
        else:
            tracking_objects_copy = tracking_objects.copy()
            for obj_id, pt2 in tracking_objects_copy.items():
                objects_exists = False
                for pt1, class_id in centroid_curr_frame:
                    dist = math.hypot(pt2[0][0] - pt1[0], pt2[0][1] - pt1[1])
                    if dist < 20:
                        tracking_objects[obj_id] = pt1, class_id
                        objects_exists = True
                        continue
                #if not objects_exists:
                 #   tracking_objects.pop(obj_id)

        #print(tracking_objects)
        #for obj_id, pt1 in tracking_objects.items():
         #   cv2.circle(frame, pt1[0], 3, (0, 255, 255), -1)
          #  cv2.putText(frame, str(obj_id)+' '+str(pt1[1]), pt1[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
            #if track:
            #objs.append([pt1[0], pt1[1]])
        #print(objs)
        centroid_prev_frame = centroid_curr_frame.copy()


    #####################
    ##        BEV      ##
    #####################
    # Define the source points (region of interest) in the original image
    x1, y1 = 10, 720  # Top-left point
    x2, y2 = 530, 400  # Top-right point
    x3, y3 = 840, 400  # Bottom-right point
    x4, y4 = 1270, 720  # Bottom-left point
    src_points = np.float32([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
    # Draw the source points on the image (in red)
    #cv2.polylines(frame, [src_points.astype(int)], isClosed=True, color=(0, 0, 255), thickness=2)

    # # Define the destination points (desired output perspective)
    u1, v1 = 370, 720  # Top-left point
    u2, v2 = 0+150, 0  # Top-right point
    u3, v3 = 1280-150, 0  # Bottom-right point
    u4, v4 = 900, 720  # Bottom-left point
    dst_points = np.float32([[u1, v1], [u2, v2], [u3, v3], [u4, v4]])
    # # Draw the destination points on the image (in blue)
    #cv2.polylines(frame, [dst_points.astype(int)], isClosed=True, color=(255, 0, 0), thickness=2)

    # perspectivs plot and objs
    transformed_image_with_centroids, persObjs_ = plot_object_bev(transformed_image_with_centroids, src_points ,dst_points , objs)
    #print("persoObjs")
    #print(persObjs_)
    ### plot objs overlays
    for persObj_ in persObjs_:
        simObjs = simulate_object(transformed_image_to_sim, persObj_[1], persObj_[0][0], persObj_[0][1])
        #print(simObjs)
    # Add the car_img overlay to the simulated image
    simulated_image = add_myCar_overlay(simObjs)

    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference Time: {inference_time} seconds")


    #videoOut.write(simulated_image)
    #videoOut.write(combined_image_horizontal)
    
    # Display the simulated image and frame

    # Resize images to have a common height
    #common_height = min(frame.shape[0], simulated_image.shape[0], transformed_image_with_centroids.shape[0])
    #frame_resized = cv2.resize(frame, (int(frame.shape[1] * (common_height / frame.shape[0])), common_height))
    #simulated_image_resized = cv2.resize(simulated_image, (int(simulated_image.shape[1] * (common_height / simulated_image.shape[0])), common_height))
    #transformed_image_with_centroids_resized = cv2.resize(transformed_image_with_centroids, (int(transformed_image_with_centroids.shape[1] * (common_height / transformed_image_with_centroids.shape[0])), common_height))

    common_height = min(frame.shape[0], simulated_image.shape[0])
    frame_resized = cv2.resize(frame, (int(frame.shape[1] * (common_height / frame.shape[0])), common_height))
    simulated_image_resized = cv2.resize(simulated_image, (int(simulated_image.shape[1] * (common_height / simulated_image.shape[0])), common_height))
    #transformed_image_with_centroids_resized = cv2.resize(transformed_image_with_centroids, (int(transformed_image_with_centroids.shape[1] * (common_height / transformed_image_with_centroids.shape[0])), common_height))

    
    # Combine images horizontally
    combined_image_horizontal = np.hstack(( simulated_image_resized,frame_resized))

    # Display the combined image
    cv2.imshow("Combined Images", combined_image_horizontal)
    videoOut.write(combined_image_horizontal)


    #cv2.imshow("Video", frame)
    
    #cv2.imshow("Simulated Objects", simulated_image)
    
    #cv2.imshow('Transformed Frame', transformed_image_with_centroids)
    
    # cv2.imwrite('test.jpg', simulated_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
video.release()
videoOut.release()
cv2.destroyAllWindows()
