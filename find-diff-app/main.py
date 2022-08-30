from pathlib import Path

from skimage import io, util
import skimage
# なぜか知らないがRGBAで読み込まれる
from skimage.color import rgba2rgb, rgb2gray
from skimage.feature import (match_descriptors, corner_harris,
                             corner_peaks, ORB, plot_matches)
import matplotlib.pyplot as plt

from skimage.util import compare_images

base = Path(__file__).parent


def src_target(img):
    # 真ん中で分ける
    height, width, color = img.shape
    w = int(width/2)
    src = util.crop(img, ((0, 0), (0, w), (0, 0)))
    target = util.crop(img, ((0, 0), (w, 0), (0, 0)))

    return src, target


def feat_orb(src, target, n_keypoints):
    """
    https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.ORB
    detector_extractor1.keypoints[matches[:, 0]]
    """
    orb = skimage.feature.ORB(n_keypoints=n_keypoints)
    #orb2 = skimage.feature.ORB(n_keypoints=n_keypoints)
    orb.detect_and_extract(src)
    keypoints1 = orb.keypoints
    descriptors1 = orb.descriptors
    orb.detect_and_extract(target)
    keypoints2 = orb.keypoints
    descriptors2 = orb.descriptors
    matches = skimage.feature.match_descriptors(
        descriptors1, descriptors2)

    return matches, keypoints1, keypoints2


def match(src, target):
    """
   2枚の画像の位置合わせを行う。 
    """

    img_src = io.imread(base / "data/src.png")
    img_target = io.imread(base / "data/target.png")

    matches, keypoints1, keypoints2 = feat_orb(
        rgb2gray(rgba2rgb(img_src)), rgb2gray(rgba2rgb(img_target)),
        n_keypoints=1000
    )

    fig = plt.figure()
    ax = fig.add_subplot()
    plot_matches(ax, img_src, img_target, keypoints1, keypoints2, matches)
    ax.axis('off')
    ax.set_title("src vs. target")

    plt.savefig(base/"data/matches.png")


def save_fig(img, name="img.png"):
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.axis('off')
    ax.imshow(img)

    plt.savefig(base / name)


def to_gray(img):
    return rgb2gray(rgba2rgb(img))


def find_diff(src, target):
    """
   差分で間違い部分を強調する
    """
    diff_img = compare_images(src, target, method='diff')

    save_fig(diff_img, name="data/diff.png")


def main():
    img = io.imread(base / 'data/AF65252C-879B-43EE-908F-DF05DC7EA278.png')
    src, target = src_target(img)
    io.imsave(base / "data/src.png", src)
    io.imsave(base / "data/target.png", target)

    img_src = io.imread(base / "data/src.png")
    img_target = io.imread(base / "data/target.png")
    find_diff(to_gray(img_src), to_gray(img_target))


if __name__ == '__main__':
    main()
