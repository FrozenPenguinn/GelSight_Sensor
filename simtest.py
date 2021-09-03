import numpy as np
import matplotlib.pyplot as plt

# camera parameters
mm2pix = 10.9 # 1mm = 10.9pix

def load_img(img_name):
    img = np.load(img_name)
    img_reshape = img[0,:,:,:]
    return img_reshape

def load_depth(depth_name):
    depth = np.load(depth_name)
    depth_reshape = depth[0,:,:]
    return depth_reshape

def show(img):
    plt.imshow(img)
    plt.colorbar()
    plt.show()
    return

def contrast(img1, img2):
    diff_img = np.zeros_like(img1)
    diff_img = (img1.astype(int) - img2.astype(int))
    return diff_img

def depth2grad(depth):
    grad = np.gradient(depth) # dR[0][:,:] gives p, dR[1][:,:] gives q
    return grad[0]

if __name__ == "__main__":
    '''
    # check images
    bg_img = load_img("./img/2_color.npy")
    show(bg_img)
    ob_img = load_img("./img/3_color.npy")
    show(ob_img)
    #diff_img = contrast(ob_img, bg_img)
    #show(diff_img)
    '''
    # check depth and grads
    img = load_img("./sim_img/1_color.npy")
    show(img)
    depth = load_depth("./sim_img/1_depth.npy") * 1000 * mm2pix
    show(depth)
    grad = depth2grad(depth)
    show(grad)
    #np.save("grad.npy", grad)
    print("done")
