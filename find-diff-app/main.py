from pathlib import Path

from skimage import io, util


def src_target(img):
    # mannakade wakeru
    height, width, color = img.shape
    w = int(width/2)
    src = util.crop(img, ((0, 0), (0, w), (0, 0)))
    target = util.crop(img, ((0, 0), (w, 0), (0, 0)))

    return src, target


def main():
    base = Path(__file__).parent
    img = io.imread(base / 'data/AF65252C-879B-43EE-908F-DF05DC7EA278.png')
    src, target = src_target(img)
    io.imsave(base / "data/src.png", src)
    io.imsave(base / "data/target.png", target)


if __name__ == '__main__':
    main()
