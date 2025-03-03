import matplotlib.pyplot as plt
import os

def plot_cell_mask(smooth_MIP_zyxin_img_corrected, smooth_MIP_pax_img_corrected,new_cell_mask_zyxin,new_cell_mask_pfak,label_cell_seg,cellmask_image_overlay,movie_panels_plot_output_dir,filename,flag_run_all,newmap):
  
    fig, ax = plt.subplots(3,2, figsize=(3,5), dpi=256, facecolor='w', edgecolor='k')
    ax[0,0].imshow(smooth_MIP_zyxin_img_corrected, cmap=plt.cm.gray,vmax=130,vmin=80)
    ax[0,0].axis('off')
    ax[0,1].imshow(smooth_MIP_pax_img_corrected, cmap=plt.cm.gray,vmax=130,vmin=80)
    ax[0,1].axis('off')

    ax[1,0].imshow(new_cell_mask_zyxin, cmap=newmap,vmax=4001,vmin=0,interpolation='None')
    ax[1,0].axis('off')
    ax[1,1].imshow(new_cell_mask_pfak, cmap=newmap,vmax=4001,vmin=0,interpolation='None')
    ax[1,1].axis('off')
    
    ax[2,0].imshow(label_cell_seg, cmap=newmap,vmax=4001,vmin=0,interpolation='None')
    ax[2,0].axis('off')
    ax[2,1].imshow(cellmask_image_overlay)
    ax[2,1].axis('off')
    plt.savefig((movie_panels_plot_output_dir+'/cellmask_'+filename+'.png'))    
    if flag_run_all > 0:
          plt.close(fig) 
