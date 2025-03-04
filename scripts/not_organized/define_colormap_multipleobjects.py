def define_colormap_multipleobjects(cm_map_type, number_limits):

    # build the colormap with iterative tab10
    colormap_single = cm.get_cmap(cm_map_type)
    colormap_multiple = colormap_single

    for i in range(number_limits/colormap_multiple.colors.shape[0]):
        colormap_multiple.colors = np.concatenate([colormap_multiple.colors,colormap_single.colors],axis=0)    

    colormap_multiple.colors = np.concatenate([0,0,0,1],colormap_multiple.colors,axis=0)    
    colormap_multiple.colors = colormap_multiple.colors[0:number_limits+1,:]