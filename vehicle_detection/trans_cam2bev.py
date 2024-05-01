import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, '..')
#from val_data_functions import Custom_ValData
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import cv2
from transweather_model import Transweather
import warnings
import math
import ssl

from PIL import Image

import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from val_data_functions import ValData,Custom_ValData,MyValData
#from utils import validation, validation_val,custom_valildation_val,custom_save_image
import os
import numpy as np
import random
from transweather_model import Transweather
from PIL import Image
from torchvision.transforms import Compose, ToTensor,Normalize
#python3 test_object_og.py -exp_name Transweather_weights

warnings.filterwarnings("ignore", category=UserWarning)

# Disable SSL certificate verification
ssl_context = ssl._create_unverified_context()

# Set SSL certificate file path
os.environ['SSL_CERT_FILE'] = 'cacert.pem'

# Load the YOLOv5 model
#model_path = 'yolov5s.pt'
#device = 'cpu'
#model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
#model.to(device)

model_path = '../yolov5s.pt'  # Specify the path to yolov5s.pt
device = 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
#model = torch.hub.load('yolov5s.pt', 'custom', path=model_path)
model.to(device)

# Function to suppress Intel MKL warnings
def suppress_mkl_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', type=str)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
args = parser.parse_args()

val_batch_size = args.val_batch_size
exp_name = args.exp_name

# Set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #random.seed(seed)
    print('Seed:\t{}'.format(seed))

# --- Set category-specific hyper-parameters  --- #
val_data_dir = '../data/radiate/rain_2_0/zed_left'

# --- Validation data loader --- #
# val_filename = 'image_paths.txt'
# val_data_loader = DataLoader(Custom_ValData(val_data_dir, val_filename), batch_size=val_batch_size, shuffle=False,
#                              num_workers=0)

# --- Define the network --- #
net = Transweather().cpu()
net = nn.DataParallel(net)

# --- Load the network weight --- #
net.load_state_dict(torch.load('../TransWeather_weights/best', map_location=torch.device('cpu')))

width, height = 1280, 720
#videoOut = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width * 2, height))  # Updated video dimensions



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

    object_img = cv2.imread(f'../assets/box.png', cv2.IMREAD_UNCHANGED)
    
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
    overlay_img = cv2.imread('../assets/MyCar.png', cv2.IMREAD_UNCHANGED)
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


# --- Use the evaluation model in testing --- #
net.eval()
category = "testmodel"
frame_count = 0
centroid_prev_frame = []
tracking_objects = {}
tracking_id = 0

print('--- Testing starts! ---')

# Process each image in the validation dataset
img_names = os.listdir(val_data_dir)
for img_name in img_names:
    #print(img_name)
    #img = os.path.join(val_data_dir,img_name)
    start_time_total = time.time()
    input_img = Image.open(os.path.join(val_data_dir,img_name))
    og_input_img = cv2.imread(os.path.join(val_data_dir,img_name))
    og_input_img = cv2.resize(og_input_img, (width, height))
    
    #og_input_img = og_input_img.resize(input_img, (width, height))
    # gt_img = Image.open(self.val_data_dir + gt_name)

    # Resizing image in the multiple of 16"
    wd_new,ht_new = input_img.size
    if ht_new>wd_new and ht_new>1024:
        wd_new = int(np.ceil(wd_new*1024/ht_new))
        ht_new = 1024
    elif ht_new<=wd_new and wd_new>1024:
        ht_new = int(np.ceil(ht_new*1024/wd_new))
        wd_new = 1024
    wd_new = int(16*np.ceil(wd_new/16.0))
    ht_new = int(16*np.ceil(ht_new/16.0))
    input_img = input_img.resize((wd_new,ht_new),Image.LANCZOS)
    # gt_img = gt_img.resize((wd_new, ht_new), Image.LANCZOS)

    # --- Transform to tensor --- #
    transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_gt = Compose([ToTensor()])
    input_im = transform_input(input_img)
    input_im = input_im.to(device)
    input_im = input_im.unsqueeze(0)
    with torch.no_grad():
        output_image = net(input_im)
    # pred_image = net(input_im)
    # #output_image = torch.split(pred_image, 1, dim=0)
    # pred_image = pred_image.cpu().numpy()
    
     
    # input_image_np = input_im.squeeze().permute(1, 2, 0).cpu().numpy()
    # output_image_np = output_image.squeeze().permute(1, 2, 0).cpu().numpy()
    
   
    input_image_np = input_im.squeeze().permute(1, 2, 0).cpu().numpy()
    output_image_np = output_image.squeeze().permute(1, 2, 0).cpu().numpy()
    print("output image shape",output_image_np.shape)
    # mean = (0.5, 0.5, 0.5)
    # std = (0.5, 0.5, 0.5)

    # # Unnormalize the output image
    # unnormalized_output = output_image_np # Create a copy to avoid modifying the original tensor
    # for i in range(3):  # Assuming it's a RGB image
    #     unnormalized_output[ :, :, i] = unnormalized_output[ :, :, i] * std[i] + mean[i]
    # output_image_np = unnormalized_output


    # # Convert back to numpy array if needed
    # unnormalized_output = unnormalized_output.numpy()
    #print("output image np",output_image_np)


        # Convert images to uint8
    input_image_np = (input_image_np * 255).astype(np.uint8)
    output_image_np = (output_image_np * 255).astype(np.uint8)
    output_image_np = cv2.cvtColor(output_image_np, cv2.COLOR_BGR2RGB ) 
    # output_image_np = (output_image_np * 255).astype(np.uint8)

    
    input_image_np= cv2.resize(input_image_np, (width, height))
    frame = cv2.resize(output_image_np, (width, height))



    frame_count += 1
    #if not success:
     #   break

    # Perform object detection on the frame
    results = model(frame, size=320)
    detections = results.pred[0]
    #end_time = time.time()
    #inference_time = end_time - start_time
    #print(f"Inference Time: {inference_time} seconds")

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


    
    

    common_height = min(frame.shape[0], simulated_image.shape[0],input_image_np.shape[0])
    frame_resized = cv2.resize(frame, (int(frame.shape[1] * (common_height / frame.shape[0])), common_height))
    simulated_image_resized = cv2.resize(simulated_image, (int(simulated_image.shape[1] * (common_height / simulated_image.shape[0])), common_height))
    input_image_np_resized = cv2.resize(input_image_np, (int(input_image_np.shape[1] * (common_height / input_image_np.shape[0])), common_height))
    #transformed_image_with_centroids_resized = cv2.resize(transformed_image_with_centroids, (int(transformed_image_with_centroids.shape[1] * (common_height / transformed_image_with_centroids.shape[0])), common_height))

    
    # Combine images horizontally
    combined_image_horizontal = np.hstack(( og_input_img, simulated_image_resized,frame_resized))

    # Display the combined image
    cv2.imshow("Combined Images", combined_image_horizontal)

    end_time_total = time.time()

    # Calculate the total time
    total_time = end_time_total - start_time_total
    # Perform perspective transformation
    #print(f"Total time for processing all images: {total_time} seconds")
    

    # Display the transformed image
    # Display input and output images using OpenCV
    #cv2.imshow('Input Image', input_image_np)
    
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    #custom_save_image(pred_image, img_name, exp_name,category)
# for idx, data in enumerate(val_data_loader):
#     input_image, target_image = data
#     input_image = input_image.cpu()
#     suppress_mkl_warnings()

#     start_time_total = time.time()
    
#     # Forward pass
#     with torch.no_grad():
#         output_image = net(input_image)
    
#     input_image_np = input_image.squeeze().permute(1, 2, 0).cpu().numpy()
#     output_image_np = output_image.squeeze().permute(1, 2, 0).cpu().numpy()
    
#     # Convert images to uint8
#     input_image_np = (input_image_np * 255).astype(np.uint8)
#     output_image_np = (output_image_np * 255).astype(np.uint8)

    
#     #frame= output_image_np


def special(camera_image):
    model_path = '../yolov5s.pt'  # Specify the path to yolov5s.pt
    device = 'cpu'
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    #model = torch.hub.load('yolov5s.pt', 'custom', path=model_path)
    model.to(device)

    # Function to suppress Intel MKL warnings
    def suppress_mkl_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)


    # --- Parse hyper-parameters  --- #
    parser = argparse.ArgumentParser(description='Hyper-parameters for network')
    parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
    parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', type=str)
    parser.add_argument('-seed', help='set random seed', default=19, type=int)
    args = parser.parse_args()

    val_batch_size = 1
    

    # Set seed
    seed = args.seed
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        #random.seed(seed)
        print('Seed:\t{}'.format(seed))

    # --- Set category-specific hyper-parameters  --- #
    val_data_dir = '../data/radiate/city_1_1/zed_left'

    # --- Validation data loader --- #
    # val_filename = 'image_paths.txt'
    # val_data_loader = DataLoader(Custom_ValData(val_data_dir, val_filename), batch_size=val_batch_size, shuffle=False,
    #                              num_workers=0)

    # --- Define the network --- #
    net = Transweather().cpu()
    net = nn.DataParallel(net)

    # --- Load the network weight --- #
    net.load_state_dict(torch.load('../TransWeather_weights/best', map_location=torch.device('cpu')))

    width, height = 1280, 720
    #videoOut = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width * 2, height))  # Updated video dimensions



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

        object_img = cv2.imread(f'../assets/box.png', cv2.IMREAD_UNCHANGED)
        
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
        overlay_img = cv2.imread('../assets/MyCar.png', cv2.IMREAD_UNCHANGED)
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


    # --- Use the evaluation model in testing --- #
    net.eval()
    category = "testmodel"
    frame_count = 0
    centroid_prev_frame = []
    tracking_objects = {}
    tracking_id = 0

    print('--- Testing starts! ---')

    # Process each image in the validation dataset
    
        #print(img_name)
        #img = os.path.join(val_data_dir,img_name)
    start_time_total = time.time()
    input_img = camera_image
    og_input_img = camera_image
    og_input_img = cv2.resize(og_input_img, (width, height))
    
    #og_input_img = og_input_img.resize(input_img, (width, height))
    # gt_img = Image.open(self.val_data_dir + gt_name)

    # Resizing image in the multiple of 16"
    wd_new,ht_new = input_img.size
    if ht_new>wd_new and ht_new>1024:
        wd_new = int(np.ceil(wd_new*1024/ht_new))
        ht_new = 1024
    elif ht_new<=wd_new and wd_new>1024:
        ht_new = int(np.ceil(ht_new*1024/wd_new))
        wd_new = 1024
    wd_new = int(16*np.ceil(wd_new/16.0))
    ht_new = int(16*np.ceil(ht_new/16.0))
    input_img = input_img.resize((wd_new,ht_new),Image.LANCZOS)
    # gt_img = gt_img.resize((wd_new, ht_new), Image.LANCZOS)

    # --- Transform to tensor --- #
    transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_gt = Compose([ToTensor()])
    input_im = transform_input(input_img)
    input_im = input_im.to(device)
    input_im = input_im.unsqueeze(0)
    with torch.no_grad():
        output_image = net(input_im)
    # pred_image = net(input_im)
    # #output_image = torch.split(pred_image, 1, dim=0)
    # pred_image = pred_image.cpu().numpy()
    
    
    # input_image_np = input_im.squeeze().permute(1, 2, 0).cpu().numpy()
    # output_image_np = output_image.squeeze().permute(1, 2, 0).cpu().numpy()
    

    input_image_np = input_im.squeeze().permute(1, 2, 0).cpu().numpy()
    output_image_np = output_image.squeeze().permute(1, 2, 0).cpu().numpy()
    print("output image shape",output_image_np.shape)
    # mean = (0.5, 0.5, 0.5)
    # std = (0.5, 0.5, 0.5)

    # # Unnormalize the output image
    # unnormalized_output = output_image_np # Create a copy to avoid modifying the original tensor
    # for i in range(3):  # Assuming it's a RGB image
    #     unnormalized_output[ :, :, i] = unnormalized_output[ :, :, i] * std[i] + mean[i]
    # output_image_np = unnormalized_output


    # # Convert back to numpy array if needed
    # unnormalized_output = unnormalized_output.numpy()
    #print("output image np",output_image_np)


        # Convert images to uint8
    input_image_np = (input_image_np * 255).astype(np.uint8)
    output_image_np = (output_image_np * 255).astype(np.uint8)
    output_image_np = cv2.cvtColor(output_image_np, cv2.COLOR_BGR2RGB ) 
    # output_image_np = (output_image_np * 255).astype(np.uint8)

    
    input_image_np= cv2.resize(input_image_np, (width, height))
    frame = cv2.resize(output_image_np, (width, height))



    frame_count += 1
    #if not success:
    #   break

    # Perform object detection on the frame
    results = model(frame, size=320)
    detections = results.pred[0]
    #end_time = time.time()
    #inference_time = end_time - start_time
    #print(f"Inference Time: {inference_time} seconds")

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


    
    

    common_height = min(frame.shape[0], simulated_image.shape[0],input_image_np.shape[0])
    frame_resized = cv2.resize(frame, (int(frame.shape[1] * (common_height / frame.shape[0])), common_height))
    simulated_image_resized = cv2.resize(simulated_image, (int(simulated_image.shape[1] * (common_height / simulated_image.shape[0])), common_height))
    input_image_np_resized = cv2.resize(input_image_np, (int(input_image_np.shape[1] * (common_height / input_image_np.shape[0])), common_height))
    #transformed_image_with_centroids_resized = cv2.resize(transformed_image_with_centroids, (int(transformed_image_with_centroids.shape[1] * (common_height / transformed_image_with_centroids.shape[0])), common_height))

    
    # Combine images horizontally
    combined_image_horizontal = np.hstack(( og_input_img, simulated_image_resized,frame_resized))

    # Display the combined image
    cv2.imshow("Combined Images", combined_image_horizontal)

    end_time_total = time.time()

    # Calculate the total time
    total_time = end_time_total - start_time_total
    # Perform perspective transformation
    #print(f"Total time for processing all images: {total_time} seconds")
    

    # Display the transformed image
    # Display input and output images using OpenCV
    #cv2.imshow('Input Image', input_image_np)
    
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()



        #custom_save_image(pred_image, img_name, exp_name,category)
    # for idx, data in enumerate(val_data_loader):
    #     input_image, target_image = data
    #     input_image = input_image.cpu()
    #     suppress_mkl_warnings()

    #     start_time_total = time.time()
        
    #     # Forward pass
    #     with torch.no_grad():
    #         output_image = net(input_image)
        
    #     input_image_np = input_image.squeeze().permute(1, 2, 0).cpu().numpy()
    #     output_image_np = output_image.squeeze().permute(1, 2, 0).cpu().numpy()
        
    #     # Convert images to uint8
    #     input_image_np = (input_image_np * 255).astype(np.uint8)
    #     output_image_np = (output_image_np * 255).astype(np.uint8)

        
    #     #frame= output_image_np

