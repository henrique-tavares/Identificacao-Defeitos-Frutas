from os import path, walk, mkdir, listdir
from math import ceil
from time import time

from skimage import io, transform, color, exposure, util, filters, morphology, segmentation, draw, measure
import numpy as np
from matplotlib import pyplot as plt
from cv2 import erode
from tqdm import tqdm

src = path.join(path.curdir, "raw-images", "abius")  # Image source folder
dest = path.join(path.curdir, "databases", "abius")  # Processed images output folder

debugging_mode = True  # Debbuging images

contour_mode = "color"  # Use 'color' (slower but more accurate) or 'binary' (faster but less accurate)

try:
    mkdir(path.join(path.curdir, dest))
    print(f"Output folder: '{dest}' successfully created!", end="\n\n")
except:
    print(f"Output folder: '{dest}' already exists.", end="\n\n")

dest_length = sum(len(files) for _, _, files in walk(dest))

for index, filename in tqdm(
    zip(range(dest_length, dest_length + len(listdir(src))), listdir(src)),
    total=len(listdir(src)),
    bar_format=" {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}] ",
):
    start = time()

    img = io.imread(path.join(src, filename))
    IMG_SIZE = 512  # 512x512

    if img.shape[0] > img.shape[1]:
        img = transform.rotate(img, 90, resize=True)

    new_scale = IMG_SIZE / img.shape[0]

    rescaled_img = transform.rescale(img, new_scale, multichannel=True, anti_aliasing=True)

    cropped_img = rescaled_img[
        :, int((rescaled_img.shape[1] - IMG_SIZE) / 2) : int(IMG_SIZE + (rescaled_img.shape[1] - IMG_SIZE) / 2)
    ]

    cropped_img = util.img_as_float(cropped_img)
    cropped_img = exposure.adjust_gamma(cropped_img)
    cropped_img = exposure.rescale_intensity(cropped_img)
    cropped_img = filters.median(cropped_img)

    if contour_mode == "binary":
        grayscale_img = color.rgb2gray(cropped_img)

        seed = np.copy(grayscale_img)
        seed[1:-1, 1:-1] = grayscale_img.max()
        mask = grayscale_img

        eroded = morphology.reconstruction(seed, mask, method="erosion")

        no_bg_img = grayscale_img - eroded

        inv_no_bg_img = util.invert(no_bg_img)
        inv_no_bg_img = exposure.rescale_intensity(inv_no_bg_img)
        inv_no_bg_img = exposure.equalize_adapthist(inv_no_bg_img)
        inv_no_bg_img = filters.median(inv_no_bg_img, morphology.disk(10))

        pre_preocessed_img = inv_no_bg_img

    elif contour_mode == "color":
        pre_preocessed_img = filters.median(cropped_img, morphology.ball(10))

    s = np.linspace(0, 2 * np.pi, 500)
    r = (IMG_SIZE / 2) + (IMG_SIZE / 2 - 5) * np.sin(s)
    c = (IMG_SIZE / 2) + (IMG_SIZE / 2 - 5) * np.cos(s)
    init = np.array([r, c]).T

    contour = segmentation.active_contour(
        pre_preocessed_img, init, alpha=0.1, beta=0.1, coordinates="rc", max_iterations=5000
    )

    mask = draw.polygon2mask((IMG_SIZE, IMG_SIZE), contour)
    mask = filters.median(mask, morphology.disk(10))

    perimeter = measure.perimeter(mask)

    kernel = np.ones((5, 5), np.uint8)
    iterations = ceil(perimeter * 0.007)
    reduced_mask = erode(util.img_as_ubyte(mask), kernel, iterations=iterations)

    processed_img = cropped_img * reduced_mask[:, :, np.newaxis]
    processed_img = exposure.adjust_gamma(processed_img)
    processed_img = exposure.rescale_intensity(processed_img)
    processed_img = exposure.equalize_adapthist(processed_img)

    finish = time()

    if debugging_mode:
        fig, axes = plt.subplots(2, 3)
        ax = axes.flatten()

        ax[0].imshow(cropped_img)
        ax[0].set_axis_off()

        ax[1].imshow(pre_preocessed_img, cmap="gray")
        ax[1].set_axis_off()

        ax[2].imshow(cropped_img)
        ax[2].plot(init[:, 1], init[:, 0], "--r", lw=2)
        ax[2].plot(contour[:, 1], contour[:, 0], "-b", lw=2)
        ax[2].set_axis_off()

        ax[3].imshow(mask, cmap="gray")
        ax[3].set_axis_off()

        ax[4].imshow(reduced_mask, cmap="gray")
        ax[4].set_axis_off()

        ax[5].imshow(processed_img)
        ax[5].set_axis_off()

        plt.show()

        tqdm.write(f"{filename} processed within {(finish - start):.1f}s")

    else:
        io.imsave(path.join(dest, f"img_{index:04}.png"), util.img_as_ubyte(processed_img))
        tqdm.write(f"{filename} processed into img_{index:04}.png successfully within {(finish - start):.1f}s")
