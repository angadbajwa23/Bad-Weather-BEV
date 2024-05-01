import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import argparse 

# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# radiate sdk
import sys
sys.path.insert(0, '..')
import radiate
import time
# path to the sequence
root_path = '../data/radiate/'
sequence_name = 'night_1_2'

network = 'faster_rcnn_R_50_FPN_3x'
setting = 'good_and_bad_weather_radar'

# time (s) to retrieve next frame
dt = 0.25

parser = argparse.ArgumentParser(description="A simple command-line tool")
# parser.add_argument('-s', '--sensor', action='store_true', help='Enable verbose mode',default='radar')
 # Add arguments
parser.add_argument('-s', '--sensor', action='store_true', help='Enable verbose mode')
parser.add_argument('sensor', metavar='sensor', type=str, default='radar', help='Sensor type (default: radar)')


args = parser.parse_args()

# load sequence
seq = radiate.Sequence(os.path.join(root_path, sequence_name), config_file='../config/config.yaml')

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(os.path.join('test','config' , network + '.yaml'))
cfg.MODEL.DEVICE = 'cpu'
if args.sensor == 'radar':
    cfg.MODEL.WEIGHTS = os.path.join('AV_radar_only_resnet50_rcnn.pth')
    print("loaded radar only model weights")
elif args.sensor == 'lidar':
    cfg.MODEL.WEIGHTS = os.path.join('lidar_model_final.pth')
    print("loaded lidar only model weights")
elif args.sensor == 'fusion':
    cfg.MODEL.WEIGHTS = os.path.join('corrected-fused-model.pth')
    print("loaded radar+lidar model weights")
else:
    print("Incorrect sensor choice")

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (vehicle)
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]

predictor = DefaultPredictor(cfg)

for t in np.arange(seq.init_timestamp, seq.end_timestamp, dt):
    
    output = seq.get_from_timestamp(t)
    seq.vis_all(output)
    if output != {}:
        start_time = time.time()
        radar = output['sensors']['radar_cartesian']
        camera = output['sensors']['camera_right_rect']
        lidar = output['sensors']['lidar_bev_image']

        if args.sensor == 'radar':
            input = radar
        elif args.sensor == 'lidar':
            input = lidar
        elif args.sensor == 'fusion':
            input = radar+lidar

        predictions = predictor(radar)
    
        predictions = predictions["instances"].to("cpu")
        boxes = predictions.pred_boxes 
        end_time = time.time()
        print("Time taken for this frame: ", end_time-start_time )
        objects = []

        for box in boxes:
            if cfg.MODEL.PROPOSAL_GENERATOR.NAME == 'RRPN':
                bb, angle = box.numpy()[:4], box.numpy()[4]        
            else:
                bb, angle = box.numpy(), 0   
                bb[2] = bb[2] - bb[0]
                bb[3] = bb[3] - bb[1]
            objects.append({'bbox': {'position': bb, 'rotation': angle}, 'class_name': 'vehicle'})
            
        radar = seq.vis(radar, objects, color=(255,0,0))
        lidar = seq.vis(lidar, objects, color=(255,0,0))
        bboxes_cam = seq.project_bboxes_to_camera(objects,
                                                seq.calib.right_cam_mat,
                                                seq.calib.RadarToRight)
        # camera = seq.vis_3d_bbox_cam(camera, bboxes_cam)
        camera = seq.vis_bbox_cam(camera, bboxes_cam)

        # cv2.imshow('our predictions', radar)
        # cv2.imshow('camera_right_rect', camera)
       
        # Resize images to have a common height
        common_height = min(radar.shape[0], camera.shape[0],lidar.shape[0])
        radar_resized = cv2.resize(radar, (int(radar.shape[1] * (common_height / radar.shape[0])), common_height))
        camera_resized = cv2.resize(camera, (int(camera.shape[1] * (common_height / camera.shape[0])), common_height))
        lidar_resized = cv2.resize(lidar, (int(lidar.shape[1] * (common_height / lidar.shape[0])), common_height))
        #transformed_image_with_centroids_resized = cv2.resize(transformed_image_with_centroids, (int(transformed_image_with_centroids.shape[1] * (common_height / transformed_image_with_centroids.shape[0])), common_height))

        # Combine images horizontally
        combined_image_horizontal = np.hstack(( radar_resized,camera_resized,lidar_resized))

        # Display the combined image
        cv2.imshow("Prediction Images", combined_image_horizontal)
        #videoOut.write(combined_image_horizontal)


        #cv2.imshow("Video", frame)
        
        #cv2.imshow("Simulated Objects", simulated_image)
        
        #cv2.imshow('Transformed Frame', transformed_image_with_centroids)
        
        # cv2.imwrite('test.jpg', simulated_image)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        

