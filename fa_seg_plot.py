# function for pax and zyxin segmentation
# note: for zyxin and paxillin data together
# data from Patrick Oaks group
def fa_seg_plot(struct_img0, filename, cell_image_path, cell_plot_path, cell_seg_path):
    org_mask = struct_img0[0,:,:]
    input_zyxin_img = struct_img0[1,:,:]
    input_pax_img = struct_img0[2,:,:]

    ### intensity normalization
    intensity_scaling_param = [5,20]
    norm_zyxin_img = intensity_normalization(input_zyxin_img, scaling_param=intensity_scaling_param)
    norm_pax_img = intensity_normalization(input_pax_img, scaling_param=intensity_scaling_param)

    # smoothing with edge preserving smoothing 
    smooth_zyxin_img = gaussian_filter(norm_zyxin_img,sigma=1,mode='nearest',truncate=3)
    smooth_pax_img = gaussian_filter(norm_pax_img,sigma=1,mode='nearest',truncate=3)

    new_cell_mask = smooth_zyxin_img>0.02
    new_cell_mask = remove_small_objects(new_cell_mask>0, min_size=20, connectivity=1, in_place=False)

    ## PARAMETERS for vesselness filter step ##
    f2_param = [[1, 0.3]]
    pax_seg = filament_2d_wrapper(smooth_pax_img, f2_param)

    f2_param = [[1, 0.4]]
    zyxin_seg = filament_2d_wrapper(smooth_zyxin_img, f2_param)


    zyxin_seg = zyxin_seg*new_cell_mask
    pax_seg = pax_seg*new_cell_mask

    label_zyxin_seg = label(zyxin_seg*new_cell_mask)
    regionprops_zyxin = regionprops(label_zyxin_seg)

    label_pax_seg = label(pax_seg*new_cell_mask)
    regionprops_pax = regionprops(label_pax_seg)


    pax_image_label_overlay = label2rgb(label_pax_seg, image=smooth_pax_img*0, bg_label=0)
    zyxin_image_label_overlay = label2rgb(label_zyxin_seg, image=smooth_zyxin_img*0, bg_label=0)
    # label_zyxin_seg[label_zyxin_seg==0]= np.nan

    fig, ax = plt.subplots(2, 4, figsize=(8,4), dpi=144, facecolor='w', edgecolor='k')
    ax[0,0].imshow(input_zyxin_img, cmap=plt.cm.gray,vmax=400,vmin=0)
    ax[0,0].axis('off')
    ax[0,1].imshow(smooth_zyxin_img, cmap=plt.cm.gray,vmax=0.3,vmin=0)
    ax[0,1].axis('off')
    ax[0,2].imshow(zyxin_seg, cmap=plt.cm.gray,vmax=0.1,vmin=0)
    ax[0,2].axis('off')
    ax[0,3].imshow(zyxin_image_label_overlay/(zyxin_image_label_overlay.max()))
    ax[0,3].axis('off')

    ax[1,0].imshow(input_pax_img, cmap=plt.cm.gray,vmax=150,vmin=0)
    ax[1,0].axis('off')
    ax[1,1].imshow(smooth_pax_img, cmap=plt.cm.gray,vmax=0.3,vmin=0)
    ax[1,1].axis('off')
    ax[1,2].imshow(pax_seg, cmap=plt.cm.gray,vmax=0.1,vmin=0)
    ax[1,2].axis('off')
    ax[1,3].imshow(pax_image_label_overlay/(pax_image_label_overlay.max()))
    ax[1,3].axis('off')

    plt.savefig(os.path.join(cell_plot_path,filename[:-4]+'org_sm_ves_seg.png'))
    
    imsave(os.path.join(cell_seg_path, 'zyxin_seg_'+filename[:-4] +'.tif'),zyxin_seg.astype(np.uint8))   
    imsave(os.path.join(cell_seg_path, 'pax_seg_'+filename[:-4] +'.tif'),pax_seg.astype(np.uint8))   
    imsave(os.path.join(cell_seg_path, 'zyxin_seglabel_'+filename[:-4] +'.tif'),label_zyxin_seg.astype(np.uint16))   
    imsave(os.path.join(cell_seg_path, 'pax_seglabel_'+filename[:-4] +'.tif'),label_pax_seg.astype(np.uint16))   
    imsave(os.path.join(cell_plot_path, 'zyxin_segrgb_'+filename[:-4] +'.tif'),zyxin_image_label_overlay.astype(np.uint8))   
    imsave(os.path.join(cell_plot_path, 'pax_segrgb_'+filename[:-4] +'.tif'),pax_image_label_overlay.astype(np.uint8))   
    
     

    dump('regionprops_pax',os.path.join(cell_seg_path, 'zyxin_regionpros_'+filename[:-4]))
    # data = load(data_filename_memmap, mmap_mode='r')

    plt.close(fig) 

