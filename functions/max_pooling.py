import numpy as np

def max_pooling( input_image:np.ndarray, pool_size = 2 ):
    # Reduces the image resolution

    iheigth,iwidth = input_image.shape

    oheigth = iheigth // pool_size
    owidth = iwidth // pool_size

    output_image = np.zeros( (oheigth,owidth) )

    for i in range( oheigth ):
        for j in range( owidth ):
            region = input_image[ i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size ]

            # trades a pool_size x pool_size image partition into its max value
            output_image[i,j] = np.max(region)

    return output_image
