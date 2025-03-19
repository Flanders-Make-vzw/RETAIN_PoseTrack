import numpy as np
import cv2
import os
import json
import scipy
import sys
sys.path.append('/home/track/aic_cpp')
import aic_cpp
from scipy.spatial.transform import Rotation as R
class Camera():
    def __init__(self,calib_data):
        self.camera_serial = calib_data["camera_serial"]

        self.project_mat = np.array(calib_data["camera projection matrix"])
        self.homo_mat = np.array(calib_data["homography matrix"])

        self.homo_inv = np.linalg.inv(self.homo_mat)
        self.project_inv = scipy.linalg.pinv(self.project_mat)

        # Calculate camera position using pseudo-inverse
        R = self.project_mat[:3, :3]  # Extract rotation matrix
        t = np.zeros(3)  # Initialize translation vector
        self.pos = -np.linalg.inv(R) @ t  # Calculate camera center

        self.homo_feet = self.homo_mat.copy()
        self.homo_feet[:, -1] = self.homo_feet[:, -1] + self.project_mat[:, 2] * 0.15  # z=0.15
        self.homo_feet_inv = np.linalg.inv(self.homo_feet)
        self.idx_int = calib_data["idx"]

    @staticmethod
    def from_colruyt_calib(calib_data, idx):
        intrinsics = calib_data["intrinsics"]["camera_matrix"]
        fx = intrinsics["fx"]
        fy = intrinsics["fy"]
        cx = intrinsics["cx"]
        cy = intrinsics["cy"]
        s  = intrinsics["s"]

        # Intrinsic Camera Matrix (K)
        K = np.array([
            [fx, s, cx],
            [0, fy, cy],
            [0,  0,  1]
        ])

        # extrinsics
        extrinsics = calib_data["extrinsics"]
        extrinsics_rotation = extrinsics["rotation"]
        w,x,y, z = extrinsics_rotation["quaternion_w"],extrinsics_rotation["quaternion_x"],extrinsics_rotation["quaternion_y"], extrinsics_rotation["quaternion_z"]
        rotation = R.from_quat([x,y, z, w]) #
        R_mat = rotation.as_matrix()
        t = np.array([[extrinsics["translation"]["x"]],
                      [extrinsics["translation"]["y"]],
                      [extrinsics["translation"]["z"]]])
        # Extrinsic matrix [R|t]
        RT = np.hstack((R_mat, t))

        # Projection matrix P
        P = K @ RT
        
        # For points on the plane z=0, the extrinsic matrix reduces to [r1, r2, t]
        # where r1 and r2 are the first two columns of R
        r1 = R_mat[:, 0]
        r2 = R_mat[:, 1]

        # Stack r1, r2, and t to form a 3x3 matrix for the homography
        H_extrinsic = np.column_stack((r1, r2, t.flatten()))

        # Compute the homography matrix H
        H = K @ H_extrinsic

        processed_calib_data = {
            "camera projection matrix": P,
            "homography matrix": H,
            "idx": idx,
            "camera_serial": calib_data["camera_serial"],
        }
        camera = Camera(processed_calib_data)
        return camera


    def to_dict(self):
        return {
            "camera_serial": self.camera_serial,
            "camera projection matrix": self.project_mat.tolist(),
            "homography matrix": self.homo_mat.tolist(),
            "idx": self.idx_int,
        }

def cross(R,V):
    h = [R[1] * V[2] - R[2] * V[1],
         R[2] * V[0] - R[0] * V[2],
         R[0] * V[1] - R[1] * V[0]]
    return h

# def Point2LineDist(p_3d, pos, ray):
#     return np.linalg.norm(cross((p_3d-pos),ray))
def Point2LineDist(p_3d, pos, ray):
    return np.linalg.norm(np.cross(p_3d-pos,ray), axis=-1)

# def Point2LineDist(p_3d, ray_world_normalized):
#     #return np.linalg.norm(cross(p_3d,ray_world))
#     return np.dot(np.concatenate(p_3d,np.array([1])),ray_world_normalized)

# def Line2LineDist(rayA,rayB):
#     pA = np.array(-rayA[-1]/(rayA[0]+1e-10), 0, 0)
#     pB = np.array(-rayB[-1]/(rayB[0]+1e-10), 0, 0)

#     return Line2LineDist(pA, rayA[:-1], pB, rayB[:-1])


def Line2LineDist(pA, rayA, pB, rayB):
    if np.abs(np.dot(rayA, rayB)) > (1 - (1e-5))* np.linalg.norm(rayA, axis=-1) * np.linalg.norm(rayB, axis=-1):  #quasi vertical
        return Point2LineDist(pA, pB, rayA)
    else:
        rayCP =  np.cross(rayA,rayB)
        return np.abs((pA-pB).dot(rayCP / np.linalg.norm(rayCP,axis=-1), axis=-1))

def Line2LineDist_norm(pA, rayA, pB, rayB):
    rayCP = np.cross(rayA, rayB, axis=-1)
    rayCP_norm = np.linalg.norm(rayCP, axis=-1) + 1e-6
    return np.abs(np.sum((pA-pB) * (rayCP / rayCP_norm[:, None]), -1))
    return np.where(
        rayCP_norm < 1e-5,
        Point2LineDist(pA, pB, rayA),
        np.abs(np.sum((pA-pB) * (rayCP / rayCP_norm[:, None]), -1))
    )
    if np.abs(np.dot(rayA, rayB)) > (1 - (1e-5)):  #quasi parallel
        return Point2LineDist(pA, pB, rayA)
    else:
        rayCP = np.cross(rayA,rayB, axis=-1)
        return np.abs(np.sum((pA-pB) * (rayCP / np.linalg.norm(rayCP,axis=-1)), -1))
    
def epipolar_3d_score(pA, rayA, pB, rayB, alpha_epi):
    dist = Line2LineDist(pA, rayA, pB, rayB)
    return 1- dist/alpha_epi

def epipolar_3d_score_norm(pA, rayA, pB, rayB, alpha_epi):
    dist = Line2LineDist_norm(pA, rayA, pB, rayB)
    return 1- dist/alpha_epi

# import aic_cpp
# print(dir(aic_cpp))

epipolar_3d_score_norm = aic_cpp.epipolar_3d_score_norm

# def epipolar_3d_score(rayA, rayB, alpha_epi):
#     dist = Line2LineDist(rayA, rayB)
#     return 1- dist/alpha_epi