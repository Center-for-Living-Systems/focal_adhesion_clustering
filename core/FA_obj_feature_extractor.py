import numpy as np
import pandas as pd
from typing import Optional, List, Union
from skimage.measure import label, regionprops
from skimage.morphology import binary_opening, binary_dilation, disk
from skimage.filters import sobel_v, sobel_h
from scipy.ndimage import distance_transform_cdt
from utils.pre_processing_utils import intensity_normalization

class FA_obj_feature_extractor:
    def __init__(
        self, 
        pax_seg: Optional[np.ndarray] = None, 
        input_multich_img: Optional[np.ndarray] = None, 
        new_cell_mask: Optional[np.ndarray] = None, 
        filename: str = "unknown", 
        filenameID: int = 0, 
        time_point: int = 0, 
        pixel_size: float = 0.07, 
        intensity_scaling_param: List[int] = [10, 40], 
        major_fa_ch: int = 0, 
        minor_fa_ch: int = 1
    ):
        self.pax_seg: np.ndarray = pax_seg if pax_seg is not None else np.zeros((100, 100), dtype=np.uint8)
        self.new_cell_mask: np.ndarray = new_cell_mask if new_cell_mask is not None else np.zeros((100, 100), dtype=bool)
        self.major_fa_ch: int = major_fa_ch
        self.minor_fa_ch: int = minor_fa_ch
        self.intensity_scaling_param: List[int] = intensity_scaling_param
        self.input_multich_img: np.ndarray = (
            input_multich_img if input_multich_img is not None else np.zeros((2, 100, 100), dtype=np.float32)
        )
        self.filename: str = filename
        self.filenameID: int = filenameID
        self.time_point: int = time_point
        self.pixel_size: float = pixel_size
        self.prop_df_pax: pd.DataFrame = pd.DataFrame()

    
    def extract_features(self):
        norm_zyxin_img = self.input_multich_img[self.minor_fa_ch]
        input_pax_img =  self.input_multich_img[self.major_fa_ch]

        norm_zyxin_img = intensity_normalization(norm_zyxin_img,self.intensity_scaling_param)
        input_pax_img = intensity_normalization(input_pax_img,self.intensity_scaling_param)

        # Label segmentations and calculate the region properties
        label_pax_seg = label(self.pax_seg)
        
        regionprops_pax = regionprops(label_pax_seg, intensity_image=input_pax_img)

        distance_taxicab = distance_transform_cdt(self.new_cell_mask, metric="taxicab")
        
        local_orientation = calculate_local_orientation(self.new_cell_mask)

        
        # Initialize feature arrays
        for iL in range(label_pax_seg.max()):  
            # get eh zyxin intensity using the mask
            zyxin_intensity_set = norm_zyxin_img[label_pax_seg == iL]

            # get the centroid of this obj
            obj_X_iL = regionprops_pax[iL].centroid[0]
            obj_Y_iL = regionprops_pax[iL].centroid[1]

            # get the distance to cell edge at centroid
            cell_edge_dist = distance_taxicab[int(obj_X_iL), int(obj_Y_iL)]
            
            # get the orientation obteined from the cell shape, at the centroid of this obj
            cell_edge_orient = local_orientation[int(obj_X_iL), int(obj_Y_iL)]
            
            # get the difference of these two orientation
            diff_orient = cell_edge_orient - regionprops_pax[iL].orientation
            
            # normalize into -pi/2 ~ pi/2, process: add pi/2, warp to 0 ~ pi and then subtract pi/2
            diff_orient = (diff_orient + np.pi / 2) % np.pi - np.pi / 2
            
            # then take the abs of the angle
            diff_orient = np.abs(diff_orient)   
            
            # Append extracted features to DataFrame
            s = pd.Series([
                self.filename, self.filenameID, self.time_point, self.pixel_size,
                regionprops_pax[iL].area, regionprops_pax[iL].bbox_area, regionprops_pax[iL].convex_area,
                regionprops_pax[iL].eccentricity, regionprops_pax[iL].equivalent_diameter, regionprops_pax[iL].euler_number,
                regionprops_pax[iL].extent, regionprops_pax[iL].filled_area, regionprops_pax[iL].label,
                regionprops_pax[iL].major_axis_length, regionprops_pax[iL].max_intensity, regionprops_pax[iL].mean_intensity,
                regionprops_pax[iL].min_intensity, regionprops_pax[iL].minor_axis_length, regionprops_pax[iL].orientation,
                regionprops_pax[iL].perimeter, regionprops_pax[iL].solidity, 
                cell_edge_dist, cell_edge_orient, diff_orient, zyxin_intensity_set.mean()
            ], index=[
                'filename', 'cell_ID', 'time_point', 'pixel_size', 'area', 'bbox_area', 'convex_area', 'eccentricity', 'equivalent_diameter', 'euler_number',
                'extent', 'filled_area', 'label', 'major_axis_length', 'max_intensity', 'mean_intensity',
                'min_intensity', 'minor_axis_length', 'orientation', 'perimeter', 'solidity',
                'cell_edge_dist', 'cell_edge_orient', 'diff_orient', 'zyxin_int'
            ])
            
            self.prop_df_pax = self.prop_df_pax.append(s, ignore_index=True)
        
        return [self.prop_df_pax, local_orientation]

def calculate_local_orientation(input_mask): 
    for_orent_mask = binary_opening(input_mask, disk(11))
    bigger_for_orent_mask = binary_dilation(for_orent_mask, disk(11))
    bigger_for_orent_distance_taxicab = distance_transform_cdt(bigger_for_orent_mask, metric="taxicab")

    # obtain cell edge orientation based on gradient of the distance map    
    n_v = sobel_v(bigger_for_orent_distance_taxicab)
    n_h = sobel_h(bigger_for_orent_distance_taxicab)
    # the gradients are really small, make them reasonable values
    for_plot_max = 5/min(n_v.max(), n_h.max())
    n_v = n_v*for_plot_max
    n_h = n_h*for_plot_max
            
    # convert the directions into orientation angels   
    local_orientation = np.arctan2(n_v,n_h)

    return local_orientation
