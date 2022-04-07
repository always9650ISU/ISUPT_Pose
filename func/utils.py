import re
import numpy as np 
import math
import cv2

def angle(v1, v2):
 
    x1,y1 = v1
    x2,y2 = v2
    dot = x1*x2+y1*y2
    det = x1*y2-y1*x2
    theta = np.arctan2(det, dot)
    theta = theta if theta>0 else 2*np.pi+theta
    
    return math.degrees(theta)

def consine_angle(vector_a, vector_b):

    vector_dot =  np.dot(vector_a, vector_b).sum() 
    vector_a_distance = np.square(vector_a).sum() ** 0.5
    vector_b_distance = np.square(vector_b).sum() ** 0.5
    vector_distance_dot = vector_a_distance * vector_b_distance
    cross = vector_dot / vector_distance_dot
    cosTheta = math.acos(cross)
    theta = math.degrees(cosTheta)

    return int(theta)


def Direction(vector):
    y  = vector[1] 
    if y > 0:
        direction =  -1
    else:
        direction = 1
    
    return direction 

def Line_dritection(point_target, point_a, point_b):
    # get plane params a,b (y = ax+b) 
    x = np.array([[point_a[0],1],[point_b[0],1]])
    y = np.array([point_a[1], point_b[1]])
    a, b = np.linalg.solve(x, y)
    y_on_plane = a * point_target[0] + b

    if point_target[1] > y_on_plane:
        direction = -1
    else:
        direction = 1

    return direction 


class BodyTheta():
    
    

    def __init__(self, img, keypoints, imgName=None):

        self.img = img
        self.Neck = keypoints[1]
        self.RShoulder = keypoints[2]
        self.RElbow = keypoints[3]
        self.RWrist = keypoints[4]
        self.LShoulder = keypoints[5]
        self.LElbow = keypoints[6]
        self.LWrist = keypoints[7]
        self.MidHip = keypoints[8]
        self.person_inclination = consine_angle(self.MidHip - self.Neck, (1000, 0))
        
        self.r = 200
        self.start_angle = 0
        self.thickness = 30
        self.txtorg = (200, 3000)
        self.txt_spacing = 200
        self.font = cv2.FONT_HERSHEY_COMPLEX 
        self.fontScale = 5
        self.txtthickness = 10
        
        if imgName.split('_')[0] == '20220224':
            self.txtorg = (60, 120)
            self.txt_spacing = 100
            self.fontScale = 2
            self.txtthickness = 5
        elif imgName.split('_')[0] == '20220302':
            self.txtorg = (60, 120)
            self.txt_spacing = 100
            self.fontScale = 2
            self.txtthickness = 5

        self.linetype = cv2.LINE_AA
        self.color_maps = [(255,0,0), 
                            (0,255, 0),
                            (0,0,255), 
                            (255,255, 0), 
                            (255, 0, 255),
                            (0, 255, 255), 
                            ()]
        self.horizont = (100, 0)

    def RElbowr_theta(self, draw=False):

        centerCoordinates = self.RShoulder
        mainPlane = self.MidHip - self.Neck
        brancePlane = self.RElbow - self.RShoulder

        theta = consine_angle(mainPlane, brancePlane)
        direction = Direction(brancePlane)
        angle = self.person_inclination

        line_rate = np.linalg.norm(brancePlane) / np.linalg.norm(mainPlane)
        line_src = centerCoordinates
        line_dst = centerCoordinates + (mainPlane) * line_rate

        if draw:
            cv2.ellipse(self.img, 
                        (int(centerCoordinates[0]),int(centerCoordinates[1])),
                        (self.r, self.r), 
                        angle,
                        self.start_angle,
                        theta, 
                        self.color_maps[0], 
                        self.thickness)
            
            cv2.line(self.img,
                    (int(line_src[0]), int(line_src[1])),
                    (int(line_dst[0]), int(line_dst[1])),
                    self.color_maps[0],
                    5)

            cv2.putText(self.img,
                        'theta:' + str(theta),
                        self.txtorg, 
                        self.font,
                        self.fontScale,
                        self.color_maps[0],
                        self.txtthickness,
                        self.linetype)

            self.txtorg = (self.txtorg[0], self.txtorg[1] + self.txt_spacing)
            
        return theta        

    def LElbowr_theta(self, draw=False):

        centerCoordinates = self.LShoulder
        mainPlane = self.MidHip - self.Neck
        brancePlane = self.LElbow - self.LShoulder

        theta = consine_angle(mainPlane, brancePlane)
        direction = Direction(brancePlane)
        angle = self.person_inclination
        
        line_rate = np.linalg.norm(brancePlane) / np.linalg.norm(mainPlane)
        line_src = centerCoordinates
        line_dst = centerCoordinates + (mainPlane) * line_rate

        if draw:
            cv2.ellipse(self.img, 
                        (int(centerCoordinates[0]),int(centerCoordinates[1])),
                        (self.r, self.r), 
                        angle,
                        self.start_angle,
                        -1 * theta, 
                        self.color_maps[1], 
                        self.thickness)

            cv2.line(self.img,
                    (int(line_src[0]), int(line_src[1])),
                    (int(line_dst[0]), int(line_dst[1])),
                    self.color_maps[1],
                    5)

            cv2.putText(self.img,
                        'theta:' + str(theta),
                        self.txtorg, 
                        self.font,
                        self.fontScale,
                        self.color_maps[1],
                        self.txtthickness,
                        self.linetype)

            self.txtorg = (self.txtorg[0], self.txtorg[1] + self.txt_spacing)
            
        return theta        

    def RWrist_theta(self, draw=False):

        centerCoordinates = self.RElbow
        mainPlane = self.RElbow - self.RShoulder
        brancePlane = self.RWrist - self.RElbow

        theta = consine_angle(mainPlane, brancePlane)

        rsholder_direction = Direction(mainPlane)
        # relbow_direction = Direction(brancePlane)
        relbow_direction = Line_dritection(self.RWrist, self.RElbow, self.RShoulder)
        

        line_rate = np.linalg.norm(brancePlane) / np.linalg.norm(mainPlane)
        line_src = centerCoordinates
        line_dst = centerCoordinates + (mainPlane) * line_rate

        

        angle = consine_angle(mainPlane, self.horizont)
        if rsholder_direction > 0:
            angle = 360 - angle
        else:
            angle = angle
          
        theta = relbow_direction * theta
        
        

        if draw:
            cv2.ellipse(self.img, 
                        (int(centerCoordinates[0]),int(centerCoordinates[1])),
                        (self.r, self.r),
                        angle,
                        self.start_angle,
                        theta, 
                        self.color_maps[2], 
                        self.thickness)

            cv2.line(self.img,
                    (int(line_src[0]), int(line_src[1])),
                    (int(line_dst[0]), int(line_dst[1])),
                    self.color_maps[2],
                    5)

            cv2.putText(self.img,
                        'theta:' + str(abs(theta)),
                        self.txtorg, 
                        self.font,
                        self.fontScale,
                        self.color_maps[2],
                        self.txtthickness,
                        self.linetype)

            self.txtorg = (self.txtorg[0], self.txtorg[1] + self.txt_spacing)
            
        return abs(theta) 

    def LWrist_theta(self, draw=False):

        centerCoordinates = self.LElbow
        mainPlane = self.LElbow - self.LShoulder
        brancePlane = self.LWrist - self.LElbow

        theta = consine_angle(mainPlane, brancePlane)

        lsholder_direction = Direction(mainPlane)
        # lelbow_direction = Direction(brancePlane)
        lelbow_direction = Line_dritection(self.LWrist, self.LElbow, self.LShoulder)
        
        angle = consine_angle(mainPlane, self.horizont)

        line_rate = np.linalg.norm(brancePlane) / np.linalg.norm(mainPlane)
        line_src = centerCoordinates
        line_dst = centerCoordinates + (mainPlane) * line_rate

        if lsholder_direction > 0:
            angle = -1 * angle 
        
        if lelbow_direction > 0:
            theta = -1 * theta


        if draw:
            cv2.ellipse(self.img, 
                        (int(centerCoordinates[0]),int(centerCoordinates[1])),
                        (self.r, self.r),
                        angle,
                        self.start_angle,
                        theta, 
                        self.color_maps[3], 
                        self.thickness)

            cv2.line(self.img,
                    (int(line_src[0]), int(line_src[1])),
                    (int(line_dst[0]), int(line_dst[1])),
                    self.color_maps[3],
                    5)

            cv2.putText(self.img,
                        'theta:' + str(abs(theta)),
                        self.txtorg, 
                        self.font,
                        self.fontScale,
                        self.color_maps[3],
                        self.txtthickness,
                        self.linetype)

        self.txtorg = (self.txtorg[0], self.txtorg[1] + self.txt_spacing)
            
        return abs(theta)

    