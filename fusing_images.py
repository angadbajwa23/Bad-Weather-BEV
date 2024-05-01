import os 
import cv2
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import math
import yaml
from utils.calibration import Calibration
import radiate




# path to the sequence
root_path = 'data/radiate/'

#sequence_name = 'city_1_1'
for folder in os.listdir(root_path):

    if folder==".DS_Store":
        continue
    # time (s) to retrieve next frame
    dt = 0.25
    print(folder)
    sequence_name = folder

    # load sequence
    seq = radiate.Sequence(os.path.join(root_path, sequence_name))

    # play sequence
    output_folder = os.path.join(root_path,sequence_name,'fused')
    print(output_folder)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    i=0
    for t in np.arange(seq.init_timestamp, seq.end_timestamp, dt):
        output = seq.get_from_timestamp(t)
        #print(i,t)
        #seq.vis_all(output, 0)
        if (output != {}):
            #print(output)
            combination =  output['sensors']['radar_cartesian'] + output['sensors']['lidar_bev_image'] + 0
            #cv2.imshow('combination', combination)
            cv2.imwrite(os.path.join(output_folder, f'{i}.png'), combination)
            i+=1

    print(f"Finished fusing images for {folder}")
    print(f"New images created = {len(os.listdir(os.path.join(root_path,sequence_name,'fused')))}")
    print(f"Original radar images created = {len(os.listdir(os.path.join(root_path,sequence_name,'Navtech_Cartesian')))}")


#     # for t in np.arange(seq.init_timestamp, seq.end_timestamp, dt):
#     #     output = seq.get_from_timestamp(t)
#     #     print(output['sensors']['lidar_bev_image'].shape)
            
#         # if (output != {}):
#         #     #print(output)
#         #     combination =  output['sensors']['radar_cartesian'] + output['sensors']['lidar_bev_image'] + 0
#         #     #cv2.imshow('combination', combination)
#         #     cv2.imwrite(os.path.join(output_folder, f'{i}.png'), combination)
        #     i+=1

# files = os.listdir('data/radiate/tiny_foggy/fused')
# print(files)
# print(sorted(files))
