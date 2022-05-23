from pathlib import Path

from skimage import io, util
import skimage


def src_target(img):
    # mannakade wakeru
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
    orb1 = skimage.feature.ORB(n_keypoints=n_keypoints)
    orb2 = skimage.feature.ORB(n_keypoints=n_keypoints)
    de1 = orb1.detect_and_extract(src)
    de2 = orb1.detect_and_extract(target)
    matches = skimage.feature.match_descriptor(
        de1.descriptors, de2.descriptors)

    return de1, de2, mathces


def main():
    base = Path(__file__).parent
    img = io.imread(base / 'data/AF65252C-879B-43EE-908F-DF05DC7EA278.png')
    src, target = src_target(img)
    io.imsave(base / "data/src.png", src)
    io.imsave(base / "data/target.png", target)


if __name__ == '__main__':
    main()
