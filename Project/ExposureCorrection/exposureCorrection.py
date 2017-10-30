import cv2
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


class ExposureCorrection:
    def __init__(self):
        n = 1          # test case
        if n == 0:
            self.folder = 'D:/testdata/970'     # image folder
            self.N = 8                          # image number
            self.images = []
            self._read_image()
            self.overlap_width = int(self.images[0].shape[1] * 2 / 3)     # overlap width
        elif n == 1:
            self.folder = 'D:/testdata/NYT/500' # image folder
            self.N = 4                          # image number
            self.images = []
            self._read_image()
            self.overlap_width = int(self.images[0].shape[1] / 2)     # overlap width

        # stack all images
        stack_image = np.hstack((tuple(self.images)))
        cv2.imwrite(self.folder + '/statck.png', stack_image)

    def test_jump(self):
        """ exposure correction method by JUMP """
        # solve the exposure gain for each image
        overlap, intensity = self.split_image()
        func = lambda x: self._min_func(x, intensity, epsilon=0.01)
        g0 = np.ones(self.N)
        res = opt.minimize(func, g0, method='Powell', options={'xtol': 1e-8, 'disp': True})
        g = res.x
        print("g = ", g)

        # correct g so that mean(g) = 1
        g = g * 1 / np.average(g)
        print("After Scale, g = ", g)

        # correction and write to file
        image_size = self.images[0].shape
        split_col0 = self.overlap_width
        split_col1 = image_size[1] - split_col0
        for n in range(self.N):
            img = self.images[n].astype(np.float32)
            img[:, :, :-1] *= g[n]
            img[img < 0] = 0
            img[img > 255] = 255
            img = img.astype(np.uint8)
            cv2.imwrite(self.folder + '/modify_' + str(n) + '.png', img)
            # write overlap region to file
            cv2.imwrite(self.folder + '/overlap_' + str(n) + '_L.png', img[:, :split_col1, :])
            cv2.imwrite(self.folder + '/overlap_' + str(n) + '_R.png', img[:, split_col0:, :])

    @staticmethod
    def _min_func(g, intensity, epsilon=0.01):
        """ function to solve gain g """
        g_plus = np.empty_like(g)
        g_plus[:-1] = g[1:]
        g_plus[-1:] = g[:1]
        N = intensity[:, 0]
        P = np.empty_like(N)
        P[:-1] = intensity[1:, 1]
        P[-1:] = intensity[:1, 1]
        return sum((g * N - g_plus * P) ** 2.0 + epsilon * (1 - g) ** 2.0)

    def split_image(self):
        """
        split each image into left and right region, and calculate the average intensity
        """
        image_size = self.images[0].shape
        split_col0 = self.overlap_width
        split_col1 = image_size[1] - split_col0
        # row_range = np.arange(int(image_size[0] * 0.2), int(image_size[0] * 0.8))
        row_range = np.arange(int(image_size[0] * 0), int(image_size[0] * 1))

        overlap = []
        intensity = np.zeros((self.N, 2))
        for n in range(self.N):
            prev_img = self.images[(n - 1) % self.N]
            current_img = self.images[n]
            next_img = self.images[(n + 1) % self.N]

            # split image
            left = current_img[row_range, :split_col1].copy()
            right = current_img[row_range, split_col0:].copy()
            pre_right = prev_img[row_range, split_col0:].copy()
            next_left = next_img[row_range, :split_col1].copy()
            # for left region overlap with previous image
            mask = cv2.bitwise_and(left[:, :, 3], pre_right[:, :, 3])
            left = cv2.bitwise_and(left[:, :, :-1], left[:, :, :-1], mask=mask)
            intensity[n, 0] = np.sum(left[:, :, :-1]) / np.count_nonzero(mask) / 3
            # for right region overlap with next image
            mask = cv2.bitwise_and(right[:, :, 3], next_left[:, :, 3])
            right = cv2.bitwise_and(right[:, :, :-1], right[:, :, :-1], mask=mask)
            intensity[n, 1] = np.sum(right[:, :, :-1]) / np.count_nonzero(mask) / 3
            overlap.append([left, right])

            # save overlap file before modify
            cv2.imwrite(self.folder + '/overlap0_' + str(n) + '_L.png', left)
            cv2.imwrite(self.folder + '/overlap0_' + str(n) + '_R.png', right)

            # plt.figure()
            # plt.subplot(121)
            # plt.imshow(left)
            # plt.subplot(122)
            # plt.imshow(right)
            # plt.show(block=False)
            print('intensity[{0}] = ({1}, {2})'.format(n, intensity[n, 0], intensity[n, 1]))

        # plt.show(block=True)
        return overlap, intensity

    def _file_at(self, index):
        """image file name at index"""
        file = self.folder
        if index == 0:
            file += '/modelseq0_idZCAM00.png'
        else:
            file += '/modelseq' + str(index) + '_idZCAM0' + str(self.N - index) + '.png'
        return file

    def _read_image(self):
        """read all image to list"""
        for i in range(self.N):
            self.images.append(cv2.imread(self._file_at(i), cv2.IMREAD_UNCHANGED))

    def _calc_avg_intensity(self, img, guide):
        """
        calculate image average intensity only both image and guide alpha are not zero
        :param img: image
        :param guide: guide image
        :return: image without value when alpha == 0
        """
        pass


corr = ExposureCorrection()
# corr.test()
corr.test_jump()
