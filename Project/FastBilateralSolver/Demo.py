from matplotlib import *
from matplotlib.pyplot import *
from bilateralSolver import *
from cv2 import imread

rcParams['image.interpolation'] = 'nearest'
rcParams['image.cmap'] = "CMRmap"
rcParams['figure.facecolor'] = 'w'

# import image
# data_folder = os.path.abspath('D:/Download/bilateral_solver-master/data/depth_superres')
data_folder = os.path.abspath('D:/Download/bilateral_solver-master/data')
reference = imread(os.path.join(data_folder, 'reference.png'))
target = imread(os.path.join(data_folder, 'target.png'), 0)
confidence = imread(os.path.join(data_folder, 'confidence.png'), 0)
im_shape = reference.shape[:2]
assert(im_shape[0] == target.shape[0])
assert(im_shape[1] == target.shape[1])
assert(im_shape[0] == confidence.shape[0])
assert(im_shape[1] == confidence.shape[1])

# draw image
figure(figsize=(14, 20))
subplot(311)
imshow(reference)
title('reference')
subplot(312)
imshow(confidence)
title('confidence')
subplot(313)
imshow(target)
title('target')
show(block=False)

# set parameters
grid_params = {
    'sigma_luma': 8,     # brightness bandwidth
    'sigma_chroma': 8,   # color bandwidth
    'sigma_spatial': 32  # spatial bandwidth
}
bs_params = {
    'lam': 128,         # The strength of the smoothness parameter
    'A_diag_min': 1e-5, # Clamp the diagonal of the A diagonal in the Jacobi preconditioner.
    'cg_tol': 1e-5,     # The tolerance on the convergence in PCG
    'cg_maxiter': 25    # The number of PCG iterations
}

# bilateral filter
grid = BilateralGrid(reference, **grid_params)
t = target.reshape(-1, 1).astype(np.double) / (pow(2, 16) - 1)
c = confidence.reshape(-1, 1).astype(np.double) / (pow(2, 16) - 1)
tc_filt = grid.filter(t * c)
c_filt = grid.filter(c)
output_filter = (tc_filt / c_filt).reshape(im_shape)
t_filt = grid.filter(t)     # filt directly

# bilateral solver
output_solver = BilateralSolver(grid, bs_params).solve(t, c).reshape(im_shape)

# visualize the calculate data
figure(figsize=(14, 20))
subplot(231)
imshow(target)
title('target')
subplot(232)
imshow(t.reshape(im_shape))
title('t')
subplot(233)
imshow(t_filt.reshape(im_shape))
title('target filter')

subplot(234)
imshow(confidence)
title('confidence')
subplot(235)
imshow(c.reshape(im_shape))
title('c')
subplot(236)
imshow(c_filt.reshape(im_shape))
title('confidence filter')
show(block=False)

# visualize the output
figure(figsize=(14, 20))
subplot(131)
imshow(target)
title('input')
subplot(132)
imshow(output_filter)
title('bilateral filter')
subplot(133)
imshow(output_solver)
title('bilateral solver')
show()

