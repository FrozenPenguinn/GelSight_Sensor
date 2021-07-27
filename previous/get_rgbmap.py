# from gradmap build rgbmap

# reading saved gradmap
gradmap = np.load( "gradmap.npy" )

# initialize rgbmap dictionary
rgbmap = {}

for p in range(0, 601):
    for q in range(0, 601):
        
