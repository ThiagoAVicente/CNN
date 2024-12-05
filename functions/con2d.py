import numpy as np

def con2d( input_image:np.ndarray , filter:np.ndarray ):
    # Applies filter to the image
    # input_image: np.array
    # filter: np.array
    # OUTPUT is a image: np.array

    iheight, iwidth = input_image.shape
    fheight, fwidth = filter.shape

    oheight = iheight - fheight + 1
    owidth = iwidth - fwidth +1
    output_image = np.zeros( (oheight,owidth) )

    for i in range( oheight ):
        for j in range( owidth ):

            region = input_image[ i:i+fheight, j:j+fwidth ]

            # Apply filter to the image using the multiplication to the filter
            output_image[i,j] = np.sum( region * filter )
    return output_image
