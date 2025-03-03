# this code is adoptted from aics-segmentation repo
# https://github.com/AllenInstitute/aics-segmentation/tree/master/aicssegmentation/core

import numpy as np
import copy
from utils.img_utils import divide_nonzero
from utils.hessian import absolute_3d_hessian_eigenvalues

def filament_2d_wrapper(struct_img, f2_param):
    bw = np.zeros(struct_img.shape, dtype=bool)

    if len(struct_img.shape)==2:
        for fid in range(len(f2_param)):
            sigma = f2_param[fid][0]
            eigenvalues = absolute_3d_hessian_eigenvalues(struct_img, sigma=sigma, scale=True, whiteonblack=True)
            res = compute_vesselness2D(eigenvalues[1], tau=1)            
    elif len(struct_img.shape)==3:
        mip = np.amax(struct_img, axis=0)
        for fid in range(len(f2_param)):
            sigma = f2_param[fid][0]
    
            res = np.zeros_like(struct_img)
            for zz in range(struct_img.shape[0]):
                tmp = np.concatenate((struct_img[zz, :, :], mip), axis=1)
                eigenvalues = absolute_3d_hessian_eigenvalues(tmp, sigma=sigma, scale=True, whiteonblack=True)
                responce = compute_vesselness2D(eigenvalues[1], tau=1)
                res[zz, :, :struct_img.shape[2]-3] = responce[:, :struct_img.shape[2]-3]            
    return res



def vesselness2D(nd_array, sigmas, tau=0.5, whiteonblack=True):

    if not nd_array.ndim == 2:
        raise(ValueError("Only 2 dimensions is currently supported"))

    # adapted from https://github.com/scikit-image/scikit-image/blob/master/skimage/filters/_frangi.py#L74
    if np.any(np.asarray(sigmas) < 0.0):
        raise ValueError("Sigma values less than zero are not valid")

    filtered_array = np.zeros(tuple([len(sigmas), ]) + nd_array.shape)

    for i, sigma in enumerate(sigmas):
        eigenvalues = absolute_3d_hessian_eigenvalues(nd_array, sigma=sigma, scale=True, whiteonblack=True)
        # print(eigenvalues[1])
        # print(eigenvalues[2])
        filtered_array[i] = compute_vesselness2D(eigenvalues[1], tau=tau)

    return np.max(filtered_array, axis=0)

def vesselness2D_range(nd_array, scale_range=(1, 10), scale_step=2, tau=0.5, whiteonblack=True):

    if not nd_array.ndim == 2:
        raise(ValueError("Only 2 dimensions is currently supported"))

    # from https://github.com/scikit-image/scikit-image/blob/master/skimage/filters/_frangi.py#L74
    sigmas = np.arange(scale_range[0], scale_range[1], scale_step)
    if np.any(np.asarray(sigmas) < 0.0):
        raise ValueError("Sigma values less than zero are not valid")

    print(sigmas)

    filtered_array = np.zeros(sigmas.shape + nd_array.shape)

    for i, sigma in enumerate(sigmas):
        eigenvalues = absolute_3d_hessian_eigenvalues(nd_array, sigma=sigma, scale=True, whiteonblack=True)
        #print(eigenvalues[1])
        #print(eigenvalues[2])
        filtered_array[i] = compute_vesselness2D(eigenvalues[1], tau = tau)

    return np.max(filtered_array, axis=0)

def vesselnessSliceBySlice(nd_array, sigmas, tau=0.5, whiteonblack=True):

    mip = np.amax(nd_array, axis=0)
    response = np.zeros(nd_array.shape)
    for zz in range(nd_array.shape[0]):
        tmp = np.concatenate((nd_array[zz, :, :], mip), axis=1)
        tmp = vesselness2D(tmp,  sigmas=sigmas, tau=1, whiteonblack=True)
        response[zz, :, :nd_array.shape[2]-3] = tmp[:, :nd_array.shape[2]-3]

    return response


def compute_vesselness2D(eigen2, tau):

    Lambda3 = copy.copy(eigen2)
    Lambda3[np.logical_and(Lambda3<0 , Lambda3 >= (tau*Lambda3.min()))]=tau*Lambda3.min()

    response = np.multiply(np.square(eigen2),np.abs(Lambda3-eigen2))
    response = divide_nonzero(27*response, np.power(2*np.abs(eigen2)+np.abs(Lambda3-eigen2),3))

    response[np.less(eigen2, 0.5*Lambda3)]=1
    response[eigen2>=0]=0
    response[np.isinf(response)]=0

    return response
