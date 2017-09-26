from matplotlib.pyplot import np
from scipy.sparse import csr_matrix

# RGB-YUV conversion
RGB_TO_YUV = np.array([
    [0.299,     0.587,     0.114],
    [-0.168736, -0.331264,  0.5],
    [0.5,      -0.418688, -0.081312]])
YUV_TO_RGB = np.array([
    [1.0,  0.0,      1.402],
    [1.0, -0.34414, -0.71414],
    [1.0,  1.772,    0.0]])
YUV_OFFSET = np.array([0, 128.0, 128.0]).reshape(1, 1, -1)


def rgb2yuv(im):
    return np.tensordot(im, RGB_TO_YUV, ([2], [1])) + YUV_OFFSET


def yuv2rgb(im):
    return np.tensordot(im.astype(np.float) - YUV_OFFSET, YUV_TO_RGB, ([2], [1]))


# bilateral grid
MAX_VAL = 255.0


def get_valid_idx(valid, candidates):
    """Find which values are present in a list and where they are located"""
    loc = np.searchsorted(valid, candidates)
    # handle edge case where the candidate is larger than all valid values
    loc = np.clip(loc, 0, len(valid) - 1)
    # identify which values are actually present
    valid_idx = np.flatnonzero(valid[loc] == candidates)
    loc = loc[valid_idx]
    return valid_idx, loc


class BilateralGrid(object):
    def __init__(self, ref, sigma_spatial=32, sigma_luma=8, sigma_chroma=8):
        im_yuv = rgb2yuv(ref)
        # compute 5-dimensional XYLUV bilateral-space coordinates
        Iy, Ix = np.mgrid[:ref.shape[0], :ref.shape[1]]
        x_coords = (Ix / sigma_spatial).astype(np.int)
        y_coords = (Iy / sigma_spatial).astype(np.int)
        luma_coords = (im_yuv[..., 0] / sigma_luma).astype(np.int)
        chroma_coords = (im_yuv[..., 1:] / sigma_chroma).astype(np.int)
        coords = np.dstack((x_coords, y_coords, luma_coords, chroma_coords))
        coords_flat = coords.reshape(-1, coords.shape[-1])
        self.npixels, self.dim = coords_flat.shape
        # hacky "hash vector" for coordinates
        # require all scaled coordinates be < MAX_VAL
        self.hash_vec = (MAX_VAL ** np.arange(self.dim))
        # construct S and B matrix
        self._compute_factorization(coords_flat)

    def _compute_factorization(self, coords_flat):
        # hash each coordinate in grid to unique value
        hashed_coords = self._hash_coords(coords_flat)
        unique_hashes, unique_idx, idx = np.unique(hashed_coords, return_index=True, return_inverse=True)
        # identify unique set of vertices
        unique_coords = coords_flat[unique_idx]
        self.nvertices = len(unique_coords)
        # construct sparse splat matrix that maps from pixels to vertices
        self.S = csr_matrix((np.ones(self.npixels), (idx, np.arange(self.npixels))))
        # construct sparse blur matrices, note that these represent [1 0 1] blurs, excluding the central element
        self.blurs = []
        for d in range(self.dim):
            blur = 0.0
            for offset in (-1, 1):
                offset_vec = np.zeros((1, self.dim))
                offset_vec[:, d] = offset
                neighbor_hash = self._hash_coords(unique_coords + offset_vec)
                valid_coord, idx = get_valid_idx(unique_hashes, neighbor_hash)
                blur = blur + csr_matrix((np.ones((len(valid_coord),)), (valid_coord, idx)),
                                         shape=(self.nvertices, self.nvertices))
            self.blurs.append(blur)

    def _hash_coords(self, coord):
        """Hacky function to turn a coordinate into a unique value"""
        return np.dot(coord.reshape(-1, self.dim), self.hash_vec)

    def splat(self, x):
        return self.S.dot(x)

    def slice(self, y):
        return self.S.T.dot(y)

    def blur(self, x):
        """Blur a bilateral-space vector with a 1 2 1 kernel in each dimension"""
        assert(x.shape[0] == self.nvertices)
        out = 2 * self.dim * x
        for blur in self.blurs:
            out = out + blur.dot(x)
        return out

    def filter(self, x):
        """Apply bilateral filter to an input x"""
        return self.slice(self.blur(self.splat(x))) / self.slice(self.blur(self.splat(np.ones_like(x))))

