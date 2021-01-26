import sys
import os

# libs to save feature arrays
import csv
import pickle
import json
from numpy import savetxt, save, savez_compressed

import numpy as np
import cv2
from utils.general import xywh2xyxy, xyxy2xywh, plot_one_box
from scipy.spatial import distance as dist

class VoteCounter():


    def __init__(self, time, fps):

        self.critical_time = time       # critical time parameter in frames units
        self.log_frames = {"FPS": fps}

        self.voters = {}
        self.voters_count = {}


    def read_urn_coordinates(self, filepath, image, radius):
        '''
        Read file with the urn coordinates, convert them into xyxy format and compute the coordinates of urn centroid
        '''
        # Read a file with urn coordinates
        if len(filepath) == 0:
            raise ValueError('No file with urn coordinates is found in labels folder. Please provide urn coordinates!')
        with open(filepath, 'r') as f:
            xywh = [float(i) for i in f.readline().split()[1:]]
            xywh = np.array(xywh).reshape(1,len(xywh))
            xyxy = xywh2xyxy(xywh).flatten()
        f.close()

        # Find centroid coordinates
        gn_array = [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]
        xyxy_urn = np.multiply(gn_array, xyxy)
        xywh_ref = xyxy2xywh(xyxy_urn.reshape((1,4)))[0]
        centroid_urn = xywh_ref[:2].reshape(1,2)

        # Save as global
        self.xyxy_urn = xyxy_urn
        self.centroid_urn = centroid_urn
        self.critical_radius = radius*np.linalg.norm(xyxy_urn[0:2]-centroid_urn[0])


    def plot_urn_bbox(self, image):
        '''
        Plot bounding box around urns
        '''
        # URN: plot bounding box
        plot_one_box(self.xyxy_urn, image, color=[255,255,255], line_thickness=1)
    
        cv2.line(image, tuple(self.xyxy_urn[0:2].astype(int)), tuple(self.centroid_urn[0].astype(int)), (255, 255, 255), thickness=1, lineType=8)
        cv2.circle(image, tuple(self.centroid_urn[0].astype(int)), radius=1, color=(255, 255, 255), thickness=4)
        cv2.circle(image, tuple(self.centroid_urn[0].astype(int)), radius=self.critical_radius.astype(int), color=(255, 255, 255), thickness=1)


    def centroid_distance(self, track, image, color, frame):
        '''
        Find a person centroid and the Eucledian distance with urn centroid
        '''
        # Show a centroid point for each person
        centroid_obj = xyxy2xywh(np.expand_dims(track[:-1], axis=0))[0,:2]
        cv2.circle(image, tuple(centroid_obj), radius=1, color=color, thickness=4)
        centroid_obj = np.expand_dims(centroid_obj, axis=0)
    
        # VOTE: euclidean distance between the urn and person centroids
        D = dist.cdist(self.centroid_urn, centroid_obj, metric="euclidean")

        track_inside = False
        if D < self.critical_radius:
            if track[-1] not in list(self.voters.keys()):
                self.voters[track[-1]] = {'initial frame': frame, frame: {'distance': D[0,0], 'centroid_coords': centroid_obj[0,:].tolist()}}
            else:
                frame_counter = frame - self.voters[track[-1]]['initial frame']
                self.voters[track[-1]][frame] = {'distance': D[0,0], 'centroid_coords': centroid_obj[0,:].tolist()}
                if frame_counter > self.critical_time:
                    self.voters_count[track[-1]] = True
            plot_one_box(track[:-1], image, label='ID'+str(int(track[-1])), color=(0,0,255), line_thickness=1)
            track_inside = True
            #cv2.putText(image,'ID '+str(int(track[-1])), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        return track_inside

    def save_voter_trajectory(self, frame, outpath):
        '''
        Save voter's trajectory into json file
        '''
        for ids in list(self.voters.keys()):
            last_frame = list(self.voters[ids].keys())[-1]
            # remove 100 frames after the latest detection
            if frame - last_frame > 100:
                # write only the ones respecting the critical time condition
                #if voters_count[ids] == True:
                with open('track_'+str(ids)+'.json', 'w') as json_file:
                    json.dump(self.voters[ids], json_file)
                del self.voters[ids]
            #print(ids, last_frame)

    def save_features_and_crops(self, image, frame, tracks, features, outpath):
        '''
        Save features and crops into files
        '''
        for i, track in enumerate(tracks):
            track_id = track[4]
            fname_features = outpath+'/features/ID_{}'.format(track_id)
            fname_crops = outpath+'/image_crops/ID_{}'.format(track_id)
            if not os.path.exists(fname_features):
                os.mkdir(fname_features)
                os.mkdir(fname_crops)
                self.log_frames['ID_'+str(track_id)] = []

            # choose format to save feature arrays on your machine: 
            # https://machinelearningmastery.com/how-to-save-a-numpy-array-to-file-for-machine-learning/
            save_format = 'csv'
            filename = fname_features+"/feature_frame_"+str(frame)
            if save_format == 'csv':
                savetxt(filename+'.csv', features[track_id], delimiter=',')
                #data = numpy.loadtxt('data.csv', delimiter=',')
            elif save_format == 'npy':
                save(filename+'.npy', features[track_id])
                #data = numpy.load('data.npy')
            elif save_format == 'npz':
                savez_compressed(filename+'.npz', features[track_id])
                # dict_data = load('data.npz'); data = dict_data['arr_0']

            # update log file with track_id detection history
            self.log_frames['ID_'+str(track_id)].append(frame)
            # save croped image
            im_crop = image[track[1]:track[3], track[0]:track[2], :]
            cv2.imwrite(filename=fname_crops+"/image_crop_"+str(frame)+'.jpg', img=im_crop)
