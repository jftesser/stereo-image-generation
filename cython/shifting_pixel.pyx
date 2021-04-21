import cv2
import cython
import numpy as np


cpdef unsigned char [:, :, :] shift(unsigned char [:, :, :] left_img, unsigned char [:, :, :] right_img, double [:, :] depth, float deviation):
    cdef int row, col, h, w
    h = <int>(left_img.shape[0])
    w = <int>(left_img.shape[1])

    cdef float dis
    cdef int col_r

    for row in range(h):
        for col in range(w):
            dis = (1.0 - depth[row, col]) * deviation
            col_r = col - <int>dis
            if col_r >= 0:
                right_img[row, col_r] = left_img[row, col]

    return right_img


cpdef unsigned char [:, :, :] inpainting(unsigned char [:, :, :] right_fix, long long[:] rows, long long[:] cols, float deviation):
    cdef int i, j, h, w
    h = <int>(right_fix.shape[0])
    w = <int>(right_fix.shape[1])
    cdef int r_offset, l_offset, offset

    for i, j in zip(rows, cols):
        for offset in range(1, <int>deviation):
            r_offset = j + offset
            l_offset = j - offset
            if r_offset < w and not np.all(right_fix[i, r_offset] == int(0)):
                right_fix[i, j] = right_fix[i, r_offset]
                break
            if l_offset >= 0 and not np.all(right_fix[i, l_offset] == int(0)):
                right_fix[i, j] = right_fix[i, l_offset]
                break

    return right_fix


cpdef unsigned char [:, :, :] overlap(unsigned char [:, :, :] left_img, unsigned char [:, :, :] right_img, unsigned char [:, :, :] composite):
    cdef int width, height, i, j
    width = <int>(left_img.shape[1])
    height = <int>(left_img.shape[0])

    for i in range(height):
        for j in range(width):
            try:
                composite[i, j, 2] = left_img[i, j, 2]
            except IndexError:
                pass

    for i in range(height):
        for j in range(width):
            try:
                composite[i, j, 1] = right_img[i, j, 1]
                composite[i, j, 0] = right_img[i, j, 0]
            except IndexError:
                pass

    return composite
