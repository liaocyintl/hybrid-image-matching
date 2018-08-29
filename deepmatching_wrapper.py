import subprocess
import cv2
import setting
import numpy as np
import secrets
import os
from pathlib import Path


def rgb_inverse_color(image):
    height, width, _ = image.shape
    img2 = image.copy()

    for i in range(height):
        for j in range(width):
            img2[i, j] = (255 - image[i, j][0], 255 - image[i, j][1], 255 - image[i, j][2])
            # img2[i,j] = (255-image[i,j])
    return img2


def imresizecrop(path, max_size=200, matching_canny=False, min_ratio=0.25, inverse=False, blur=False):
    src = cv2.imread(str(path))

    (h, w, c) = src.shape

    if h > w:
        height = max_size
        ratio = src.shape[0] * 1.0 / height

        if w / h < min_ratio:
            width = int(height * min_ratio)
        else:
            width = int(src.shape[1] * 1.0 / ratio)

        temp_image = cv2.resize(src, (width, height))
    else:
        width = max_size
        ratio = src.shape[1] * 1.0 / width

        if h / w < min_ratio:
            height = int(width * min_ratio)
        else:
            height = int(src.shape[0] * 1.0 / ratio)

        temp_image = cv2.resize(src, (width, height))

    name = secrets.token_hex(16) + ".jpg"

    if matching_canny:
        temp_image = cv2.Canny(temp_image, 50, 100)

    if inverse:
        temp_image = rgb_inverse_color(temp_image)

    if blur:
        temp_image = cv2.GaussianBlur(temp_image, (5, 5), 0)

    cv2.imwrite(str(Path("temp/%s" % name)), temp_image)
    return name, width, height, temp_image


def match(img1_path, img2_path, inverse=False, maxsize=200):
    name1, name2 = "", ""
    try:

        name1, qw, qh, img1 = imresizecrop(img1_path, max_size=setting.MAX_WIDTH)
        name2, tw, th, img2 = imresizecrop(img2_path, max_size=setting.MAX_WIDTH, inverse=inverse)

        res = subprocess.run(
            [str(Path("deepmatching/deepmatching-static")), str(Path("temp/%s" % name1)), str(Path("temp/%s" % name2)), "-nt", str(setting.CPU_CORE)],
            stdout=subprocess.PIPE)

        os.remove(Path("temp/%s" % name1))
        os.remove(Path("temp/%s" % name2))

        result = []

        rows = res.stdout.decode("utf-8").split("\n")

        for row in rows:
            v = row.split()

            if len(v) == 6:
                result.append([int(v[0]), int(v[1]), int(v[2]), int(v[3]), float(v[4]), int(v[5])])

        return result, name1, name2, qw, qh, tw, th, img1, img2

    except Exception as err:
        print("Exception", err)
        return [], name1, name2, 0, 0, 0, 0, [], []



def parse_matches(matches):
    M = []
    for i, match in enumerate(matches):
        if i % 4 == 0:
            M.append(match)
    return M


def draw_img(src1, src2, matches, path=None):
    import random

    width_offset = src1.shape[1]

    height = src1.shape[0] + src2.shape[0]
    width = src1.shape[1] + src2.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:src1.shape[0], 0:src1.shape[1]] = src1
    output[0:src1.shape[0], src1.shape[1]:] = src1[:]

    output[src1.shape[0]:src2.shape[0] + src1.shape[0], 0:src2.shape[1]] = src2
    output[src1.shape[0]:src2.shape[0] + src2.shape[0], src2.shape[1]:] = src2[:]

    random.seed(444)
    r = lambda: random.randint(0, 255)

    for match in matches:
        color = (r(), r(), r())

        cv2.line(output, tuple(map(int, (width_offset + match[0],  match[1]))),
                 tuple(map(int, (width_offset + match[2], src1.shape[0] + match[3]))),
                 color)
        cv2.circle(output, tuple(map(int, (width_offset + match[0],  match[1]))), 1, color, 2)
        cv2.circle(output, tuple(map(int, (width_offset + match[2], src1.shape[0] + match[3]))), 1, color, 2)

    cv2.imwrite(str(path), output)

def matching_test():
    path1 = Path("deepmatching/climb1.png")
    path2 = Path("deepmatching/climb2.png")

    matches, name1, name2, qw, qh, tw, th, img1, img2 = match(path1, path2, inverse=True)

    src_pts = np.float32([[m[0], m[1]] for m in matches])
    dst_pts = np.float32([[m[2], m[3]] for m in matches])

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, setting.RANSAC_THRESHOLD)
    i = 0
    inliear = []
    for index, m in enumerate(mask):
        if np.isclose(m, 1):
            i += 1
            inliear.append(matches[index])

    path = Path("temp/test.jpg")
    draw_img(img1, img2, inliear, path)
    print("matches", len(matches))
    print("inliear", i)


if __name__ == "__main__":
    matching_test()
