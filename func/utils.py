import re
from time import thread_time
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
    try:
        cosTheta = math.acos(cross)
    except ValueError:
        print(ValueError)
        cosTheta = 0
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
    
    def __new__(cls, img, keypoints, threshold=0):
        if keypoints[8].all() == 0 or keypoints[1].all() == 0:
            print("person_inclination can\'t detection.")
            return None
        else:
            return object.__new__(cls)

    def __init__(self, img, keypoints, threshold = 0):
        
        
        h, w, c = img.shape
        
        img_ext_scale = 0.2
        img_ = np.zeros((h, w + int(w * img_ext_scale) ,c), dtype=np.uint8)
        img_[0:h, 0:w, :] = img
        self.img = img_
        self.threshold = threshold
        keypointsOrder = np.where(keypoints[:,2] < self.threshold)
        
        for i in keypointsOrder:
            keypoints[i] = 0

        self.keypoints = keypoints
        self.Neck = keypoints[1,:2]
        self.RShoulder = keypoints[2, :2]
        self.RElbow = keypoints[3, :2]
        self.RWrist = keypoints[4, :2]
        self.LShoulder = keypoints[5, :2]
        self.LElbow = keypoints[6, :2]
        self.LWrist = keypoints[7, :2]
        self.MidHip = keypoints[8, :2]
        self.RHip = keypoints[9, :2]
        self.RKnee = keypoints[10, :2]
        self.RAnkle = keypoints[11, :2]
        self.LHip = keypoints[12, :2]
        self.LKnee = keypoints[13, :2]
        self.LAnkle = keypoints[14, :2]
        self.LBigToe = keypoints[19, :2]
        self.LSmallToe = keypoints[20, :2]
        self.LHeel = keypoints[21, :2]
        self.RBigToe = keypoints[22, :2]
        self.RSmallToe = keypoints[23, :2]
        self.RHeel = keypoints[24, :2]
        self.person_inclination = consine_angle(self.MidHip - self.Neck, (1000, 0))
        
        #If  From left to right. else right to left.
        if (self.LBigToe[0] > self.LHeel[0]) and (self.LBigToe.all() != 0) and (self.LHeel.all() != 0):
            self.walkDirect = 1
        elif (self.RBigToe[0] > self.RHeel[0]) and (self.RBigToe.all() != 0) and (self.RHeel.all() != 0): 
            self.walkDirect = 1
        else : 
            self.walkDirect = -1
        
        person_sacle = np.linalg.norm(keypoints[8] - keypoints[1])
        height, width, _ = img.shape

        self.start_angle = 0
        self.font = cv2.FONT_HERSHEY_COMPLEX 
        self.r = int(person_sacle * 0.1)
        self.thickness = int(height / 200)
        self.txtorg = (30 + w, 60)#(200, 3000)
        self.txt_spacing = int(height / 20)
        self.fontScale = int(height / 1000)
        self.txtthickness = int(height / 500)
        self.auxiliaryLine_think = int(height / 500)
        if self.auxiliaryLine_think < 2:
            self.auxiliaryLine_think = 2
        
        self.linetype = cv2.LINE_AA
        self.KeypointsColorMaps = [(250,0,0),(250,50,0),(250,100,0),(250,150,0),(250,200,0),(250,250,0),
                            (200,250,0),(150,250,0),(100,2500,),(50,250,0),(0,250,0),
                            (0,250,50),(0,250,100),(0,250,150),(0,250,200),(0,250,250),
                            (0,200,250),(0,50,250),(0,100,250),(0,50,250),(0,0,250),
                            (50,0,250),(100,0,250),(150,0,250),(200,0,250),(250,0,250),]
        self.EllipseColorMaps = [(250,100,50),(250,150,50),
                                (250,100,100),(250,150,100),
                                (250,100,150),(250,150,150),
                                (250,100,200),(250,150,200),
                                (250,100,250),(250,150,250),
                                (150,250,50),(100,250,50),
                                (200,250,100),(150,250,100),
                                (0,250,100),(200,250,150),
                                (50,250,150),(0,250,150),
                                (100,250,200),(50,250,200),
                                (150,250,250),(100,250,250)]

        self.AuxiliaryLineColor = (255,100,50)

        self.horizont = (100, 0)

    def draw_keypoints(self,):
        line_thickness = 2
        r = 2

        keypoint_pairs = [
                        [0,1], 
                        [1,2], [2,3], [3,4], 
                        [1,5], [5,6], [6,7], 
                        [0,15], [15,17], 
                        [0,16], [16,18],
                        [1,8], 
                        [8,9], [9,10], [10,11],[11,24], [24,22], [23, 22], 
                        [8,12], [12,13], [13,14],[14,21], [21,19],[20,19]
                        ]
        drawed_points = []
        for num, pairs in enumerate(keypoint_pairs):
            pointA, pointB = pairs

            if self.keypoints[pointA][2] < self.threshold or self.keypoints[pointB][2] < self.threshold:
                continue

            Ax, Ay, _ = map(int, self.keypoints[pointA])
            Bx, By, _ = map(int, self.keypoints[pointB])
            cv2.line(self.img, 
                    (Ax, Ay), 
                    (Bx, By), 
                    self.KeypointsColorMaps[num],
                    line_thickness
                    )
            
            if not pointA in drawed_points:
                cv2.circle(self.img, 
                            (Ax, Ay),
                            r,
                            self.KeypointsColorMaps[num],
                            -1 )
            
            if not pointB in drawed_points:
                cv2.circle(self.img, 
                            (Bx, By),
                            r,
                            self.KeypointsColorMaps[num],
                            -1 )

    def RShoulder_theta(self, draw=False):

        if self.MidHip.all() == 0 or self.Neck.all() == 0 or self.RElbow.all() == 0 or self.RShoulder.all() == 0: 
            return "Keypoint can\'t detection. Confident less 0.5"

        centerCoordinates = self.RShoulder[:2]
        mainPlane = self.MidHip[:2] - self.Neck[:2]
        brancePlane = self.RElbow[:2] - self.RShoulder[:2]

        if mainPlane.all() == 0 or brancePlane.all() == 0:
            return "Keypoint can\'t detection."

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
                        self.EllipseColorMaps[0], 
                        self.thickness)
            
            cv2.line(self.img,
                    (int(line_src[0]), int(line_src[1])),
                    (int(line_dst[0]), int(line_dst[1])),
                    self.AuxiliaryLineColor,
                    self.auxiliaryLine_think)
            # print(thickness)
            cv2.putText(self.img,
                        'RShoulder theta:' + str(theta),
                        self.txtorg, 
                        self.font,
                        self.fontScale,
                        self.EllipseColorMaps[0],
                        self.txtthickness,
                        self.linetype)

            self.txtorg = (self.txtorg[0], self.txtorg[1] + self.txt_spacing)
            
        return theta        

    def LShoulder_theta(self, draw=False):

        if self.MidHip.all() == 0 or self.Neck.all() == 0 or self.LElbow.all() == 0 or self.LShoulder.all() == 0: 
            return "Keypoint can\'t detection. Confident less 0.5"

        centerCoordinates = self.LShoulder
        mainPlane = self.MidHip - self.Neck
        brancePlane = self.LElbow - self.LShoulder
        
        if mainPlane.all() == 0 or brancePlane.all() == 0:
            return "Keypoint can\'t detection."

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
                        self.EllipseColorMaps[1], 
                        self.thickness)

            cv2.line(self.img,
                    (int(line_src[0]), int(line_src[1])),
                    (int(line_dst[0]), int(line_dst[1])),
                    self.AuxiliaryLineColor,
                    self.auxiliaryLine_think)

            cv2.putText(self.img,
                        'LShoulder theta:' + str(theta),
                        self.txtorg, 
                        self.font,
                        self.fontScale,
                        self.EllipseColorMaps[1],
                        self.txtthickness,
                        self.linetype)

            self.txtorg = (self.txtorg[0], self.txtorg[1] + self.txt_spacing)
            
        return theta        

    def RElbow_theta(self, draw=False):

        if self.RElbow.all() == 0 or self.RShoulder.all() == 0 or self.RWrist.all() == 0: 
            return "Keypoint can\'t detection. Confident less 0.5"

        centerCoordinates = self.RElbow
        mainPlane = self.RElbow - self.RShoulder
        brancePlane = self.RWrist - self.RElbow
        
        if mainPlane.all() == 0 or brancePlane.all() == 0:
            return "Keypoint can\'t detection."

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
                        self.EllipseColorMaps[2], 
                        self.thickness)

            cv2.line(self.img,
                    (int(line_src[0]), int(line_src[1])),
                    (int(line_dst[0]), int(line_dst[1])),
                    self.AuxiliaryLineColor,
                    self.auxiliaryLine_think)

            cv2.putText(self.img,
                        'RElbow theta:' + str(abs(theta)),
                        self.txtorg, 
                        self.font,
                        self.fontScale,
                        self.EllipseColorMaps[2],
                        self.txtthickness,
                        self.linetype)

            self.txtorg = (self.txtorg[0], self.txtorg[1] + self.txt_spacing)
            
        return abs(theta) 

    def LElbow_theta(self, draw=False):
        
        if self.LElbow.all() == 0 or self.LShoulder.all() == 0 or self.LWrist.all() == 0:
            return "Keypoint can\'t detection. Confident less 0.5"

        centerCoordinates = self.LElbow
        mainPlane = self.LElbow - self.LShoulder
        brancePlane = self.LWrist - self.LElbow

        if mainPlane.all() == 0 or brancePlane.all() == 0:
            return "Keypoint can\'t detection."
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
                        self.EllipseColorMaps[3], 
                        self.thickness)

            cv2.line(self.img,
                    (int(line_src[0]), int(line_src[1])),
                    (int(line_dst[0]), int(line_dst[1])),
                    self.AuxiliaryLineColor,
                    self.auxiliaryLine_think)

            cv2.putText(self.img,
                        'LElbow theta:' + str(abs(theta)),
                        self.txtorg, 
                        self.font,
                        self.fontScale,
                        self.EllipseColorMaps[3],
                        self.txtthickness,
                        self.linetype)

        self.txtorg = (self.txtorg[0], self.txtorg[1] + self.txt_spacing)
            
        return abs(theta)
        
    def RHip_theta(self, draw=False):

        if self.MidHip.all() == 0 or self.Neck.all() == 0 or self.RKnee.all() == 0 or self.RHip.all() == 0: 
            return "Keypoint can\'t detection. Confident less 0.5"

        centerCoordinates = self.RHip
        mainPlane = self.MidHip - self.Neck
        brancePlane = self.RKnee - self.RHip

        if mainPlane.all() == 0 or brancePlane.all() == 0:
            return "Keypoint can\'t detection."

        theta = consine_angle(mainPlane, brancePlane)
        theta = theta * self.walkDirect

        line_rate = np.linalg.norm(brancePlane) / np.linalg.norm(mainPlane)
        line_src = centerCoordinates
        line_dst = centerCoordinates + (mainPlane) * line_rate

        angle = consine_angle(mainPlane, self.horizont)

        if line_dst[0] > self.RKnee[0]:
            theta = theta * -1

        
        if draw:
            cv2.ellipse(self.img, 
                        (int(centerCoordinates[0]),int(centerCoordinates[1])),
                        (self.r, self.r),
                        angle,
                        self.start_angle,
                        theta, 
                        self.EllipseColorMaps[4], 
                        self.thickness)

            cv2.line(self.img,
                    (int(line_src[0]), int(line_src[1])),
                    (int(line_dst[0]), int(line_dst[1])),
                    self.AuxiliaryLineColor,
                    self.auxiliaryLine_think)

            cv2.putText(self.img,
                        'RHip theta:' + str(abs(theta)),
                        self.txtorg, 
                        self.font,
                        self.fontScale,
                        self.EllipseColorMaps[4],
                        self.txtthickness,
                        self.linetype)

        self.txtorg = (self.txtorg[0], self.txtorg[1] + self.txt_spacing)

        return theta

    def LHip_theta(self, draw=False):
        
        if self.MidHip.all() == 0 or self.Neck.all() == 0 or self.LKnee.all() == 0 or self.LHip.all() == 0: 
            return "Keypoint can\'t detection. Confident less 0.5"

        centerCoordinates = self.LHip
        mainPlane = self.MidHip - self.Neck
        brancePlane = self.LKnee - self.LHip

        if mainPlane.all() == 0 or brancePlane.all() == 0:
            return "Keypoint can\'t detection."

        theta = consine_angle(mainPlane, brancePlane)
        theta = theta * self.walkDirect

        line_rate = np.linalg.norm(brancePlane) / np.linalg.norm(mainPlane)
        line_src = centerCoordinates
        line_dst = centerCoordinates + (mainPlane) * line_rate

        angle = consine_angle(mainPlane, self.horizont)

        if line_dst[0] > self.LKnee[0]:
            theta = theta * -1

        
        if draw:
            cv2.ellipse(self.img, 
                        (int(centerCoordinates[0]),int(centerCoordinates[1])),
                        (self.r, self.r),
                        angle,
                        self.start_angle,
                        theta, 
                        self.EllipseColorMaps[5], 
                        self.thickness)

            cv2.line(self.img,
                    (int(line_src[0]), int(line_src[1])),
                    (int(line_dst[0]), int(line_dst[1])),
                    self.AuxiliaryLineColor,
                    self.auxiliaryLine_think)

            cv2.putText(self.img,
                        'LHip theta:' + str(abs(theta)),
                        self.txtorg, 
                        self.font,
                        self.fontScale,
                        self.EllipseColorMaps[5],
                        self.txtthickness,
                        self.linetype)

        self.txtorg = (self.txtorg[0], self.txtorg[1] + self.txt_spacing)

        return theta
    
    def RKnee_theta(self, draw=False):
        
        if self.RKnee.all() == 0 or self.RHip.all() == 0 or self.RAnkle.all() == 0 or self.RKnee.all() == 0: 
            return "Keypoint can\'t detection. Confident less 0.5"

        centerCoordinates = self.RKnee
        mainPlane = self.RKnee - self.RHip
        brancePlane = self.RAnkle - self.RKnee

        if mainPlane.all() == 0 or brancePlane.all() == 0:
            return "Keypoint can\'t detection."
        
        theta = consine_angle(mainPlane, brancePlane)
        theta = theta * self.walkDirect
        
        line_rate = np.linalg.norm(brancePlane) / np.linalg.norm(mainPlane)
        line_src = centerCoordinates
        line_dst = centerCoordinates + (mainPlane) * line_rate

        angle = consine_angle(mainPlane, self.horizont)
        
        if draw:
            cv2.ellipse(self.img, 
                        (int(centerCoordinates[0]),int(centerCoordinates[1])),
                        (self.r, self.r),
                        angle,
                        self.start_angle,
                        theta, 
                        self.AuxiliaryLineColor, 
                        self.thickness)

            cv2.line(self.img,
                    (int(line_src[0]), int(line_src[1])),
                    (int(line_dst[0]), int(line_dst[1])),
                    self.KeypointsColorMaps[3],
                    self.auxiliaryLine_think)

            cv2.putText(self.img,
                        'RKnee theta:' + str(abs(theta)),
                        self.txtorg, 
                        self.font,
                        self.fontScale,
                        self.AuxiliaryLineColor,
                        self.txtthickness,
                        self.linetype)

        self.txtorg = (self.txtorg[0], self.txtorg[1] + self.txt_spacing)
        
        if self.walkDirect == -1 and (line_dst[0] < self.RAnkle[0]):
            theta = -1 * theta
        elif self.walkDirect == -1 and (line_dst[0] > self.RAnkle[0]):
            theta = theta
        elif self.walkDirect == 1 and (line_dst[0] < self.RAnkle[0]):
            theta = theta
        else:
            theta = -1 * theta

        return theta

    def LKnee_theta(self, draw=False):
        
        if self.LKnee.all() == 0 or self.LHip.all() == 0 or self.LAnkle.all() == 0 or self.LKnee.all() == 0: 
            return "Keypoint can\'t detection. Confident less 0.5"

        centerCoordinates = self.LKnee
        mainPlane = self.LKnee - self.LHip
        brancePlane = self.LAnkle - self.LKnee

        if mainPlane.all() == 0 or brancePlane.all() == 0:
            return "Keypoint can\'t detection."

        theta = consine_angle(mainPlane, brancePlane)
        # print(theta)
        theta = theta * self.walkDirect

        line_rate = np.linalg.norm(brancePlane) / np.linalg.norm(mainPlane)
        line_src = centerCoordinates
        line_dst = centerCoordinates + (mainPlane) * line_rate

        angle = consine_angle(mainPlane, self.horizont)
        
        if draw:
            cv2.ellipse(self.img, 
                        (int(centerCoordinates[0]),int(centerCoordinates[1])),
                        (self.r, self.r),
                        angle,
                        self.start_angle,
                        theta, 
                        self.EllipseColorMaps[7], 
                        self.thickness)

            cv2.line(self.img,
                    (int(line_src[0]), int(line_src[1])),
                    (int(line_dst[0]), int(line_dst[1])),
                    self.AuxiliaryLineColor,
                    self.auxiliaryLine_think)

            cv2.putText(self.img,
                        'LKnee theta:' + str(abs(theta)),
                        self.txtorg, 
                        self.font,
                        self.fontScale,
                        self.EllipseColorMaps[7],
                        self.txtthickness,
                        self.linetype)

        self.txtorg = (self.txtorg[0], self.txtorg[1] + self.txt_spacing)

        if self.walkDirect == -1 and (line_dst[0] < self.LAnkle[0]):
            theta = -1 * theta
        elif self.walkDirect == -1 and (line_dst[0] > self.LAnkle[0]):
            theta = theta
        elif self.walkDirect == 1 and (line_dst[0] < self.LAnkle[0]):
            theta = theta
        else:
            theta = -1 * theta

        return theta

    def RAnkle_theta(self, draw=False):

        if self.RAnkle.all() == 0 or self.RKnee.all() == 0 or self.RSmallToe.all() == 0 or self.RHeel.all() == 0: 
            return "Keypoint can\'t detection. Confident less 0.5"

        centerCoordinates = self.RHeel
        mainPlane = self.RAnkle - self.RKnee
        brancePlane = self.RSmallToe - self.RHeel

        if mainPlane.all() == 0 or brancePlane.all() == 0:
            return "Keypoint can\'t detection."

        theta = consine_angle(mainPlane, brancePlane) - 90
        theta = theta * self.walkDirect
        
        normal_vector = np.array((self. walkDirect* mainPlane[1], mainPlane[0]))

        line_rate = np.linalg.norm(brancePlane) / np.linalg.norm(mainPlane)
        line_src = centerCoordinates
        line_dst = centerCoordinates + (normal_vector) * line_rate

        if line_dst[1] < self.RSmallToe[1]:
            theta = -1 * theta
        else:
            theta = theta
        
        angle = consine_angle(mainPlane, self.horizont) + 90 
        
        if draw:
            cv2.ellipse(self.img, 
                        (int(centerCoordinates[0]),int(centerCoordinates[1])),
                        (self.r, self.r),
                        angle,
                        self.start_angle,
                        theta, 
                        self.EllipseColorMaps[8], 
                        self.thickness)

            cv2.line(self.img,
                    (int(line_src[0]), int(line_src[1])),
                    (int(line_dst[0]), int(line_dst[1])),
                    self.AuxiliaryLineColor,
                    self.auxiliaryLine_think)

            cv2.putText(self.img,
                        'RAnkle theta:' + str(abs(theta)),
                        self.txtorg, 
                        self.font,
                        self.fontScale,
                        self.EllipseColorMaps[8],
                        self.txtthickness,
                        self.linetype)

        self.txtorg = (self.txtorg[0], self.txtorg[1] + self.txt_spacing)

        return theta
    
    def LAnkle_theta(self, draw=False):

        if self.LAnkle.all() == 0 or self.LKnee.all() == 0 or self.LSmallToe.all() == 0 or self.LHeel.all() == 0: 
            return "Keypoint can\'t detection. Confident less 0.5"

        centerCoordinates = self.LHeel
        mainPlane = self.LAnkle - self.LKnee
        brancePlane = self.LSmallToe - self.LHeel

        if mainPlane.all() == 0 or brancePlane.all() == 0:
            return "Keypoint can\'t detection."
        
        theta = consine_angle(mainPlane, brancePlane) - 90
        theta = theta * self.walkDirect
        
        normal_vector = np.array((self.walkDirect* mainPlane[1], mainPlane[0]))

        line_rate = np.linalg.norm(brancePlane) / np.linalg.norm(mainPlane)
        line_src = centerCoordinates
        line_dst = centerCoordinates + (normal_vector) * line_rate

        if line_dst[1] < self.LSmallToe[1]:
            theta = -1 * theta
        else:
            theta = theta

        angle = consine_angle(mainPlane, self.horizont) + 90 
        
        if draw:
            cv2.ellipse(self.img, 
                        (int(centerCoordinates[0]),int(centerCoordinates[1])),
                        (self.r, self.r),
                        angle,
                        self.start_angle,
                        theta, 
                        self.EllipseColorMaps[9], 
                        self.thickness)

            cv2.line(self.img,
                    (int(line_src[0]), int(line_src[1])),
                    (int(line_dst[0]), int(line_dst[1])),
                    self.AuxiliaryLineColor,
                    self.auxiliaryLine_think)

            cv2.putText(self.img,
                        'LAnkle theta:' + str(abs(theta)),
                        self.txtorg, 
                        self.font,
                        self.fontScale,
                        self.EllipseColorMaps[9],
                        self.txtthickness,
                        self.linetype)

        self.txtorg = (self.txtorg[0], self.txtorg[1] + self.txt_spacing)

        return theta