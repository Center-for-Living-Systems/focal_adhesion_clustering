import numpy as np
# from plot_cell_mask import plot_cell_mask
from scipy.ndimage import gaussian_filter
from skimage.morphology import binary_opening, binary_closing, binary_dilation
from skimage.morphology import disk
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops, regionprops_table
from skimage.color import label2rgb
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_cdt


def zyxin_pfak_cell_mask_seg(zyxin_img,pax_img,thres_zyxin, thres_pax, close_size,movie_panels_plot_output_dir,filename,flag_run_all,newmap):
    kernel_size = zyxin_img.shape[0]
    sigma = 1
    muu = 0
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                        np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x**2+y**2)

    # Calculating Gaussian kernel
    gauss = np.exp(-((x**2+y**2) / (2.0 * sigma**2))) 
    pfak_median_background_img = gauss*25+100
    zyxin_median_background_img = gauss*8+100

    smooth_MIP_zyxin_img = gaussian_filter(zyxin_img,sigma=2,mode='nearest',truncate=3)
    smooth_MIP_pax_img = gaussian_filter(pax_img,sigma=2,mode='nearest',truncate=3)

    smooth_MIP_pax_img_corrected = smooth_MIP_pax_img*100/pfak_median_background_img
    smooth_MIP_zyxin_img_corrected = smooth_MIP_zyxin_img*100/zyxin_median_background_img

    ## low threshold to get cell masks
    new_cell_mask_zyxin = smooth_MIP_zyxin_img_corrected > thres_zyxin
    # new_cell_mask = np.logical_or(smooth_MIP_zyxin_img>103, smooth_MIP_pax_img>0.06)
    new_cell_mask_pfak = smooth_MIP_pax_img_corrected > thres_pax

    # new_cell_mask = np.logical_or(new_cell_mask_pfak, new_cell_mask_zyxin)
    new_cell_mask = new_cell_mask_zyxin

    ### remove isolated small objects (small regions due to noise)
    new_cell_mask = remove_small_objects(new_cell_mask>0, min_size=3, connectivity=1)     
    new_cell_mask = binary_closing(new_cell_mask, disk(close_size))   

    new_cell_mask = remove_small_objects(new_cell_mask>0, min_size=10, connectivity=1)    
    new_cell_mask = binary_opening(new_cell_mask, disk(3))   

    # remove holes
    new_cell_mask = ndimage.binary_fill_holes(new_cell_mask)


    new_cell_mask = remove_small_objects(new_cell_mask>0, min_size=30000, connectivity=1)    
    
    # keep the one in the center
    label_cell_seg = label(new_cell_mask)
    regionprops_cells = regionprops(label_cell_seg)
    

    if(len(regionprops_cells)>1):
        distance_array = np.zeros(len(regionprops_cells))+10000
        boarder_flag = np.zeros(len(regionprops_cells))
        
        for ci in range(len(regionprops_cells)):
            distance_array[ci] = abs(regionprops_cells[ci]['Centroid'][0]- float(new_cell_mask.shape[0])/2) +abs(regionprops_cells[ci]['Centroid'][1]- float(new_cell_mask.shape[1])/2)
            if regionprops_cells[ci]['bbox'][0] ==0 or regionprops_cells[ci]['bbox'][1] ==0:
                boarder_flag[ci]=1
            if regionprops_cells[ci]['bbox'][2] == new_cell_mask.shape[0] or regionprops_cells[ci]['bbox'][3] == new_cell_mask.shape[1]:
                boarder_flag[ci]=1
    
        sort_ind= np.argsort(distance_array)

        new_cell_mask_center  = np.zeros_like(new_cell_mask)
        first_label = regionprops_cells[sort_ind[0]]['label']
        second_label = regionprops_cells[sort_ind[1]]['label']

        new_cell_mask_center[label_cell_seg==first_label] = first_label
        # print(first_label)
        if(boarder_flag[sort_ind[1]]==0):
            new_cell_mask_center[label_cell_seg==second_label] = second_label
    else:
           new_cell_mask_center =  new_cell_mask.copy()
    # calculate the distance map

    label_cell_seg_center = label(new_cell_mask_center>0)
    display_pax = smooth_MIP_pax_img_corrected/np.percentile(smooth_MIP_pax_img_corrected[smooth_MIP_pax_img_corrected>=0],98.5)
    # display_pax = display_pax*1.2
    display_pax[display_pax>1]=1

    cellmask_image_overlay = label2rgb(label_cell_seg_center+2, image=display_pax, kind='overlay',alpha=0.1,colors=newmap.colors)
 
    fig, ax = plt.subplots(2,3, figsize=(5,3), dpi=256, facecolor='w', edgecolor='k')
    ax[0,0].imshow(smooth_MIP_zyxin_img_corrected, cmap=plt.cm.gray,vmax=130,vmin=80)
    ax[0,0].axis('off')
    ax[1,0].imshow(smooth_MIP_pax_img_corrected, cmap=plt.cm.gray,vmax=130,vmin=80)
    ax[1,0].axis('off')

    ax[0,1].imshow(new_cell_mask_zyxin, cmap=newmap,vmax=4001,vmin=0,interpolation='None')
    ax[0,1].axis('off')
    ax[1,1].imshow(new_cell_mask_pfak, cmap=newmap,vmax=4001,vmin=0,interpolation='None')
    ax[1,1].axis('off')
    
    ax[0,2].imshow(label_cell_seg, cmap=newmap,vmax=4001,vmin=0,interpolation='None')
    ax[0,2].axis('off')

    ax[1,2].imshow(smooth_MIP_pax_img_corrected, cmap=plt.cm.gray,vmax=130,vmin=80)

    for_orent_distance_taxicab = distance_transform_cdt(new_cell_mask_center, metric="taxicab")
    # for_orent_distance_taxicab[for_orent_distance_taxicab>1]=0
    X, Y = np.meshgrid(np.arange(0,label_cell_seg.shape[1]), np.arange(0,label_cell_seg.shape[0]))
    ax[1,2].contour(X, Y, for_orent_distance_taxicab,0,linewidths=0.2,colors='yellow')  
    ax[1,2].axis('off')
    plt.savefig((movie_panels_plot_output_dir+'/cellmask_display'+filename+'.png'))    
    if flag_run_all > 0:
          plt.close(fig) 

    fig1, ax1 = plt.subplots(1,1, figsize=(5,5), dpi=256, facecolor='w', edgecolor='k')
    ax1.imshow(smooth_MIP_pax_img_corrected, cmap=plt.cm.gray,vmax=130,vmin=80)
    ax1.contour(X, Y, for_orent_distance_taxicab,0,linewidths=0.5,colors='yellow')  
    ax1.axis('off')
    plt.savefig((movie_panels_plot_output_dir+'/cellmask_'+filename+'.png')) 
    if flag_run_all > 0:
        plt.close(fig1)    
    
    return [new_cell_mask_center, smooth_MIP_pax_img_corrected, smooth_MIP_zyxin_img_corrected]