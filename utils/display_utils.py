from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import os
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.morphology import binary_opening, binary_dilation
from skimage.morphology import disk
from scipy.ndimage import distance_transform_cdt
from utils.pre_processing_utils import intensity_normalization

def define_colormap_multipleobjects(cm_map_type: str = "tab10", number_limits: int = 4000) -> ListedColormap:
    """
    Generate a colormap that extends an existing colormap to accommodate a large number of segmented objects.

    Parameters:
    - cm_map_type (str, default "tab10"): Name of the colormap.
    - number_limits (int, default 4000): Number of required colors.

    Returns:
    - ListedColormap: A new colormap with `number_limits` colors.
    """
    # Retrieve the base colormap
    colormap_single = cm.get_cmap(cm_map_type)

    # Extract colors from the colormap
    base_colors = colormap_single(np.linspace(0, 1, colormap_single.N))  # Get discrete colors

    # Repeat colors to reach `number_limits`
    num_repeats = (number_limits // colormap_single.N) + 1  # Ensure enough colors
    extended_colors = np.tile(base_colors, (num_repeats, 1))  # Repeat color set

    # Add black (0,0,0,1) as the first color
    extended_colors = np.vstack([[0, 0, 0, 1], extended_colors[:number_limits]])

    # Return a new colormap with the modified colors
    return ListedColormap(extended_colors)


def display_fa_segmentation_panels(input_image, input_cell_mask, input_obj_seg,local_orientation, plot_output_dir, filename, flag_plot_save = 1, flag_run_all = 1):
    
    if flag_plot_save:
        # prepare folders
        movie_contour_output_dir = os.path.join(plot_output_dir,  'contour')
        movie_quiver_cell_output_dir = os.path.join(plot_output_dir,  'quiver_cell')
        movie_quiver_obj_output_dir = os.path.join(plot_output_dir,  'quiver_obj')
        movie_label_color_output_dir = os.path.join(plot_output_dir,  'label_color')
        movie_rgb_plot_output_dir = os.path.join(plot_output_dir,  'rgb_plot')
        movie_panels_plot_output_dir = os.path.join(plot_output_dir,  'panel_plot')

        os.makedirs(movie_contour_output_dir, exist_ok=True)
        os.makedirs(movie_quiver_cell_output_dir, exist_ok=True)
        os.makedirs(movie_quiver_obj_output_dir, exist_ok=True)
        os.makedirs(movie_label_color_output_dir, exist_ok=True)
        os.makedirs(movie_rgb_plot_output_dir, exist_ok=True)
        os.makedirs(movie_panels_plot_output_dir, exist_ok=True)
        
    # label segmentations and calculate the region properties
    label_FA_obj = label(input_obj_seg)
    regionprops_pax = regionprops(label_FA_obj,intensity_image=input_image)
    extended_colors = define_colormap_multipleobjects()

    # prepare the color map with max of the labels for display (consistent through all images)
    pax_image_label_overlay = label2rgb(label_FA_obj, image=input_image*4,kind='overlay',alpha=0.75,colors=extended_colors)

    for_orent_mask = binary_opening(input_cell_mask, disk(11))
    for_orent_distance_taxicab = distance_transform_cdt(for_orent_mask, metric="taxicab")


    # find the orentation and directional vectors for each object at the centroid        
    # initialize the vectors
    obj_X = np.zeros([label_FA_obj.max(),1])
    obj_Y = np.zeros([label_FA_obj.max(),1])
    cell_U = np.zeros([label_FA_obj.max(),1])
    cell_V = np.zeros([label_FA_obj.max(),1])
    obj_U = np.zeros([label_FA_obj.max(),1])
    obj_V = np.zeros([label_FA_obj.max(),1])
    cell_tan_U = np.zeros([label_FA_obj.max(),1])
    cell_tan_V = np.zeros([label_FA_obj.max(),1])
    
    for iL in range(label_FA_obj.max()):          

        obj_X[iL] = regionprops_pax[iL]['centroid'][0]
        obj_Y[iL] = regionprops_pax[iL]['centroid'][1]

        obj_U[iL] = np.sin(regionprops_pax[iL]['orientation'])
        obj_V[iL] = np.cos(regionprops_pax[iL]['orientation'])

        cell_edge_orient = local_orientation[int(obj_X[iL]),int(obj_Y[iL])]
        cell_tan_U[iL] = np.sin(cell_edge_orient)
        cell_tan_V[iL] = np.cos(cell_edge_orient)
        
        cell_U[iL] = n_v[int(obj_X[iL]),int(obj_Y[iL])]
        cell_V[iL] = n_h[int(obj_X[iL]),int(obj_Y[iL])]

        diff_orient = cell_edge_orient - regionprops_pax[iL]['orientation']

        if(diff_orient>np.pi/2):       diff_orient = diff_orient - np.pi
        if(diff_orient>np.pi/2):       diff_orient = diff_orient - np.pi
        if(diff_orient<-np.pi/2):      diff_orient = diff_orient + np.pi
        if(diff_orient<-np.pi/2):      diff_orient = diff_orient + np.pi

        diff_orient = np.abs(diff_orient)    
        n_v = np.cos(local_orientation)   
        n_h = np.sin(local_orientation)   

        X, Y = np.meshgrid(np.arange(0,input_cell_mask.shape[1]), np.arange(0,input_cell_mask.shape[0]))
        # build a sparse grid and only within cell mask for quivering the directions
        grid_mask = np.zeros_like(input_cell_mask)
        grid_mask[::30,::30] = input_cell_mask[::30,::30]
        to_plot_X = X[grid_mask>0]
        to_plot_Y = Y[grid_mask>0]
        to_plot_U = n_v[grid_mask>0]
        to_plot_H = n_h[grid_mask>0]


        # plot the main outputs together for quick viewing
        fig, ax = plt.subplots(2,4, figsize=(16,8), dpi=256, facecolor='w', edgecolor='k')
        ax[0,0].imshow(input_image, cmap=plt.cm.gray,vmax=0.8,vmin=0)
        ax[0,0].axis('off')
        ax[0,1].imshow(input_cell_mask, cmap=plt.cm.gray,vmax=0.3,vmin=0)
        ax[0,1].axis('off')

        # build a grid for contour plot

        ax[0,2].imshow(for_orent_distance_taxicab,cmap=plt.cm.gray)
        ax[0,2].contour(X, Y, for_orent_distance_taxicab,6,colors=('yellow','green', 'r','blue','cyan'),linewidths=0.7)
        ax[0,2].axis('off')        
    
        ax[0,3].imshow(for_orent_distance_taxicab,cmap=plt.cm.gray)
        ax[0,3].quiver(to_plot_X,to_plot_Y, -to_plot_U,to_plot_H,color='blue')
        ax[0,3].contour(X, Y, for_orent_distance_taxicab,6,linewidths=0.1)    
        ax[0,3].axis('off')


        ax[1,0].imshow(input_cell_mask, cmap=plt.cm.gray,vmax=0.1,vmin=0)
        ax[1,0].axis('off')
        
        ax[1,1].imshow(label_FA_obj, cmap=extended_colors, interpolation='none',vmax = 4001,vmin = 0)
        ax[1,1].axis('off')

        ax[1,2].imshow(pax_image_label_overlay)
        ax[1,2].axis('off')
        
        ax[1,3].imshow(input_image,cmap=plt.cm.gray,vmax=1,vmin=0)
        # quiver the orientation based on cell shape in blue
        ax[1,3].quiver(obj_Y,obj_X, -cell_U,cell_V,color='blue')
        # quiver the orientation of each object in magenta        
        ax[1,3].quiver(obj_Y,obj_X,  -obj_U,obj_V,color='m')         
        ax[1,3].axis('off')

        if flag_plot_save: 
            # save the plots and subplots for easier viewing
            plt.savefig(os.path.join(movie_panels_plot_output_dir,'panels_'+filename+'_MIP'+'_org_sm_ves_seg.png'))

            extent = ax[1,3].get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(os.path.join(movie_quiver_obj_output_dir,'quiver_obj_'+filename+'_MIP'+'.png'), bbox_inches=extent)

            extent = ax[0,2].get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(os.path.join(movie_contour_output_dir,'contours_'+filename+'_MIP'+'.png'), bbox_inches=extent)

            extent = ax[0,3].get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(os.path.join(movie_quiver_cell_output_dir,'quiver_cell_'+filename+'_MIP'+'.png'), bbox_inches=extent)

            extent = ax[1,1].get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(os.path.join(movie_label_color_output_dir,'labels_'+filename+'_MIP'+'.png'), bbox_inches=extent)

            extent = ax[1,2].get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(os.path.join(movie_rgb_plot_output_dir,'rgb_'+filename+'_MIP'+'.png'), bbox_inches=extent)
        
        if flag_run_all:
            plt.close(fig) 
    