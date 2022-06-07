from email.quoprimime import body_check
import cv2
import os
import time
import numpy as np 
from func.utils import consine_angle, angle, BodyTheta
from moviepy.editor import VideoFileClip
import json

start_time = time.time()


# img_path = './data/1.jpg'
InputDir = './data'
OutputDir = './output'
JsonDir = './keypoints_json'
ThetaDir = './theta_json'
start_time = time.time()



for FullName in os.listdir(InputDir):
    filename, extension = FullName.split('.')
    json_file = os.path.join(JsonDir, filename + '.json')
    theta_file = os.path.join(ThetaDir, filename + '.json')
    filepath = os.path.join(InputDir, FullName)
    outputpath = os.path.join(OutputDir, FullName)
    keypoints_theta = {}
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            
    except FileNotFoundError:
        continue

    print(FullName)

    if extension == 'jpg':
        
        keypoints = np.array(data['keypoints'])
        img = cv2.imread(filepath)
        body_theta = BodyTheta(img, keypoints, 0.5)
        if body_theta is None: 
            continue
        body_theta.draw_keypoints()
        RShoulder_theta = body_theta.RShoulder_theta(draw=True)
        LShoulder_theta = body_theta.LShoulder_theta(draw=True)
        RElbow_theta = body_theta.RElbow_theta(draw=True)
        LElbow_theta = body_theta.LElbow_theta(draw=True)
        RHip_theta = body_theta.RHip_theta()
        LHip_theta = body_theta.LHip_theta() 
        RKnee_theta = body_theta.RKnee_theta()
        LKnee_theta = body_theta.LKnee_theta(True)
        RAnkle_theta = body_theta.RAnkle_theta()
        LAnkle_theta = body_theta.LAnkle_theta()

        keypoints_theta  = {
            'RShoulder_theta':RShoulder_theta,
            'LShoulder_theta':LShoulder_theta,
            'RElbow_theta':RElbow_theta,
            'LElbow_theta':LElbow_theta,
            'RHip_theta':RHip_theta,
            'LHip_theta':LHip_theta,
            'RKnee_theta':RKnee_theta,
            'LKnee_theta':LKnee_theta,
            'RAnkle_theta':RAnkle_theta,
            'LAnkle_theta':LAnkle_theta,
        }

        img = body_theta.img
        cv2.imwrite(outputpath, img)

    elif extension == 'mp4' or extension == 'MOV':
        cap = cv2.VideoCapture(filepath)

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        out = cv2.VideoWriter(outputpath, fourcc, fps, (int(width + width *0.2), int(height)))

        frame_counter = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                print('end!')
                break
            try:
                keypoints = np.asarray(data[str(frame_counter)]['keypoints'], dtype=float)
            except:
                print(f'{frame_counter} keypoints can not found')
                frame_counter += 1
                out.write(frame)
                continue

            body_theta = BodyTheta(frame, keypoints[0], 0.1)
            if body_theta is None: 
                continue

            body_theta.draw_keypoints()
            RShoulder_theta = body_theta.RShoulder_theta(draw=True)
            LShoulder_theta = body_theta.LShoulder_theta(draw=True)
            RElbow_theta = body_theta.RElbow_theta(draw=True)
            LElbow_theta = body_theta.LElbow_theta(draw=True)
            RHip_theta = body_theta.RHip_theta(True)
            LHip_theta = body_theta.LHip_theta(True) 
            RKnee_theta = body_theta.RKnee_theta(True)
            LKnee_theta = body_theta.LKnee_theta(True)
            RAnkle_theta = body_theta.RAnkle_theta(draw=True)
            LAnkle_theta = body_theta.LAnkle_theta(draw=True)

            keypoints_theta[str(frame_counter)] = {
                'RShoulder_theta':RShoulder_theta,
                'LShoulder_theta':LShoulder_theta,
                'RElbow_theta':RElbow_theta,
                'LElbow_theta':LElbow_theta,
                'RHip_theta':RHip_theta,
                'LHip_theta':LHip_theta,
                'RKnee_theta':RKnee_theta,
                'LKnee_theta':LKnee_theta,
                'RAnkle_theta':RAnkle_theta,
                'LAnkle_theta':LAnkle_theta,
            }

            img = body_theta.img
            out.write(img)
            cv2.imshow('img', img)
            cv2.imwrite(os.path.join('./temp_keypoints', str(frame_counter)+'.jpg'), img)
            key = cv2.waitKey(1)
            if key == 27 or 0xFF == ord('q'):
                break 
            print(f'frame:{frame_counter}')
            frame_counter += 1
            
            
        cap.release()
        out.release()

        # audioclip = VideoFileClip(filepath).audio
        # videoclip = VideoFileClip(outputpath).set_audio(audioclip)
        # videoclip.write_videofile(outputpath, codec="libx264")

    with open(theta_file, 'w',  newline='') as f:
        json.dump(keypoints_theta, f,  indent=4)
        

    