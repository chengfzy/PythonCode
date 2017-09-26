from matplotlib import *
from matplotlib.pyplot import *
from cv2 import imread, imwrite
from FastBilateralSolver.bilateralgrid import *
from FastBilateralSolver.bilateralSolver import *
import numpy as np
import cv2
import struct

rcParams['image.interpolation'] = 'nearest'
rcParams['image.cmap'] = "CMRmap"
rcParams['figure.facecolor'] = 'w'


# read flow from data
def read_flow(file, flow):
    with open(file, 'rb') as f:
        rows = struct.unpack('i', f.read(4))[0]
        cols = struct.unpack('i', f.read(4))[0]
        # flow = np.zeros([rows, cols], dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                # data = struct.unpack('f', f.read(4))
                # print(r, c, data)
                flow[r][c] = struct.unpack('f', f.read(4))[0]

# write flow to data
def write_flow(file, flow):
    rows, cols = flow.shape
    with open(file, "wb+") as f:
        byte = struct.pack('i', rows)
        f.write(byte)
        byte = struct.pack('i', cols)
        f.write(byte)
        for r in range(rows):
            for c in range(cols):
                byte = struct.pack('f', flow[r][c])
                f.write(byte)


# flow_read = np.zeros([1, 1])
# read_flow('D:/flowdata/flow_filt.bin', flow_read)


# import image
data_folder = os.path.abspath('D:/flowdata/')
reference = imread(os.path.join(data_folder, 'overlap_7_L.png'))
im_shape = reference.shape[:2]
# target = imread(os.path.join(data_folder, 'flowLtoR_7x.png'), 0)
target = np.zeros(im_shape, dtype=np.float32)
read_flow(os.path.join(data_folder, 'flow_x.bin'), target)
confidence = imread(os.path.join(data_folder, 'confidence.png'), 0)
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
    'sigma_chroma': 50,   # color bandwidth
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

# re-scale
output_filter = (output_filter.reshape(-1, 1) * (pow(2, 16) - 1)).reshape(im_shape)
output_solver = (output_solver.reshape(-1, 1) * (pow(2, 16) - 1)).reshape(im_shape)


# temp_rgb0 = reference
# cv2.cvtColor(reference, 6, temp_rgb0)
# temp_yuv = rgb2yuv(temp_rgb0)
# temp_rgb1 = yuv2rgb(temp_yuv)


# normalize flow to 0-255
def norm_flow(flow):
    flow_size = flow.shape
    flow = flow.reshape(-1, 1).astype(np.double)
    f0 = min(flow)
    f1 = max(flow)
    return ((flow - f0) / (f1 - f0)).reshape(flow_size)


# save data
flow_filt_norm = (norm_flow(tc_filt / c_filt) * 255.0).reshape(im_shape)
flow_solver_norm = (norm_flow(output_solver) * 255.0).reshape(im_shape)
imwrite(os.path.join(data_folder, 'flow_filt.png'), flow_filt_norm)
imwrite(os.path.join(data_folder, 'flow_solver.png'), flow_solver_norm)
write_flow(os.path.join(data_folder, 'flow_filt.bin'), output_filter)
write_flow(os.path.join(data_folder, 'flow_solver.bin'), output_solver)

# flow_filt = (tc_filt / c_filt).reshape(im_shape)
# flow_filt = np.tile(flow_filt.reshape(im_shape[0], im_shape[1], -1), (1, 1, 3))
# flow_filt_rgb = yuv2rgb(flow_filt)
# flow_filt_gray = flow_filt_rgb
# cv2.cvtColor(flow_filt_rgb, 6, flow_filt_gray)
#
# write_flow(os.path.join(data_folder, 'flow_filt.bin'), flow_filt_gray)
# write_flow(os.path.join(data_folder, 'flow_solver.bin'), yuv2rgb(output_solver))


# visualize the calculate data
figure(figsize=(14, 20))
subplot(2, 3, 1)
imshow(target)
title('target')
subplot(2, 3, 2)
imshow(t.reshape(im_shape))
title('t')
subplot(2, 3, 3)
imshow(t_filt.reshape(im_shape))
title('target filter')
subplot(2, 3, 4)
imshow(confidence)
title('confidence')
subplot(2, 3, 5)
imshow(c.reshape(im_shape))
title('c')
subplot(2, 3, 6)
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






