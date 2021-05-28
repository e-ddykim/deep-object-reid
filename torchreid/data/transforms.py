from __future__ import absolute_import, division, print_function
import math
from os import stat
import random
from collections import deque
from os.path import exists, join

import cv2
import numpy as np
import torch
import re
from PIL import Image, ImageOps, ImageEnhance, ImageChops
from torchvision.transforms import (ColorJitter, Compose, Normalize, Resize,
                                    ToPILImage, ToTensor, InterpolationMode)
from torchvision.transforms import functional as F

from torchreid.utils.tools import read_image
from ..data.datasets.image.lfw import FivePointsAligner


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, input_tuple):
        if random.random() > self.p:
            return input_tuple

        img, mask = input_tuple

        img = F.hflip(img)
        mask = F.hflip(mask) if mask != '' else mask

        return img, mask


class CenterCrop(object):
    def __init__(self, margin=0, **kwargs):
        self.margin = margin

    def __call__(self, input_tuple, *args, **kwargs):
        if self.margin <= 0:
            return input_tuple

        img, mask = input_tuple
        img_width, img_height = img.size
        if img_width - 2 * self.margin < 2 or img_height - 2 * self.margin < 2:
            return input_tuple

        box = (self.margin, self.margin, img_width - self.margin, img_height - self.margin)
        img = img.crop(box)
        mask = mask.crop(box) if mask != '' else mask

        return img, mask


class RandomCrop(object):
    def __init__(self, p=0.5, scale=0.9, static=False, margin=None, target_ar=None, align_ar=False, align_center=False, **kwargs):
        self.p = p
        assert 0.0 <= self.p <= 1.0
        self.scale = scale
        assert 0.0 < self.scale < 1.0
        self.margin = margin
        self.target_ar = target_ar
        self.align_center = align_center
        self.static = static
        self.align_ar = align_ar
        if self.align_ar:
            assert self.target_ar is not None and self.target_ar > 0

    def __call__(self, input_tuple, *args, **kwargs):
        if random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple
        img_width, img_height = img.size

        if self.align_ar:
            source_ar = float(img_height) / float(img_width)
            target_ar = random.uniform(min(source_ar, self.target_ar), max(source_ar, self.target_ar))

            if target_ar < source_ar:
                max_crop_width = img_width
                max_crop_height = target_ar * max_crop_width
            else:
                max_crop_height = img_height
                max_crop_width = max_crop_height / target_ar
        else:
            max_crop_width = img_width
            max_crop_height = img_height

        if self.margin is None or self.margin <= 0:
            min_scale = self.scale
        else:
            width_rest = max(1, img_width - 2 * self.margin)
            height_rest = max(1, img_height - 2 * self.margin)

            min_width_scale = float(width_rest) / float(img_width)
            min_height_scale = float(height_rest) / float(img_height)
            min_scale = max(min_width_scale, min_height_scale)

        if self.static:
            scale = min_scale
        else:
            scale = random.uniform(min_scale, 1.0)
        crop_width = int(round(scale * max_crop_width))
        crop_height = int(round(scale * max_crop_height))

        if self.align_center:
            min_crop_width = min_scale * img_width
            min_crop_height = min_scale * img_height

            center_x = 0.5 * img_width
            center_y = 0.5 * img_height

            x_shift_range = (max(0, center_x + 0.5 * min_crop_width - crop_width),
                             min(img_width - crop_width, center_x - 0.5 * min_crop_width))
            y_shift_range = (max(0, center_y + 0.5 * min_crop_height - crop_height),
                             min(img_height - crop_height, center_y - 0.5 * min_crop_height))
        else:
            x_shift_range = 0, img_width - crop_width
            y_shift_range = 0, img_height - crop_height

        x1 = int(round(random.uniform(*x_shift_range)))
        y1 = int(round(random.uniform(*y_shift_range)))

        img = img.crop((x1, y1, x1 + crop_width, y1 + crop_height))
        mask = mask.crop((x1, y1, x1 + crop_width, y1 + crop_height)) if mask != '' else mask
        return img, mask


class RandomErasing(object):
    """Randomly erases an image patch.

    Origin: `<https://github.com/zhunzhong07/Random-Erasing>`_

    Reference:
        Zhong et al. Random Erasing Data Augmentation.

    Args:
        probability (float, optional): probability that this operation takes place.
            Default is 0.5.
        sl (float, optional): min erasing area.
        sh (float, optional): max erasing area.
        r1 (float, optional): min aspect ratio.
    """

    def __init__(self, p=0.5, sl=0.02, sh=0.4, rl=0.5, rh=2.0, fill_color=None, norm_image=True, **kwargs):
        self.probability = p
        self.sl = sl
        self.sh = sh
        self.rl = rl
        self.rh = rh
        self.norm_image = norm_image
        self.fill_color = fill_color
        if self.fill_color is not None and len(self.fill_color) in (1, 3):
            if len(self.fill_color) == 1:
                self.fill_color = [self.fill_color] * 3
        else:
            self.fill_color = None

    def __call__(self, input_tuple, *args, **kwargs):
        if random.uniform(0, 1) > self.probability:
            return input_tuple

        img, mask = input_tuple
        img_size = img.size() if self.norm_image else img.size

        for attempt in range(100):
            source_area = img_size[0] * img_size[1]
            target_area = random.uniform(self.sl, self.sh) * source_area
            aspect_ratio = random.uniform(self.rl, self.rh)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img_size[1] and h < img_size[0]:
                x1 = random.randint(0, img_size[0] - h)
                y1 = random.randint(0, img_size[1] - w)

                fill_color = self.fill_color if self.fill_color is not None else [random.randint(0, 255)] * 3
                if self.norm_image:
                    fill_color = np.array(fill_color) / 255.0

                img = img if self.norm_image else np.array(img)
                img[x1:x1 + h, y1:y1 + w, 0] = fill_color[0]
                img[x1:x1 + h, y1:y1 + w, 1] = fill_color[1]
                img[x1:x1 + h, y1:y1 + w, 2] = fill_color[2]
                img = img if self.norm_image else Image.fromarray(img)

                return img, mask

        return img, mask


class ColorAugmentation(object):
    """Randomly alters the intensities of RGB channels.

    Reference:
        Krizhevsky et al. ImageNet Classification with Deep ConvolutionalNeural
        Networks. NIPS 2012.

    Args:
        p (float, optional): probability that this operation takes place.
            Default is 0.5.
    """

    def __init__(self, p=0.5, **kwargs):
        self.p = p
        self.eig_vec = torch.Tensor([[0.4009, 0.7192, -0.5675],
                                     [-0.8140, -0.0045, -0.5808],
                                     [0.4203, -0.6948, -0.5836],
                                     ])
        self.eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])

    def __call__(self, input_tuple, *args, **kwargs):
        if random.uniform(0, 1) > self.p:
            return input_tuple

        tensor, mask = input_tuple

        alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)

        return tensor, mask


class RandomPatch(object):
    """Random patch data augmentation.

    There is a patch pool that stores randomly extracted pathces from person images.

    For each input image, RandomPatch
        1) extracts a random patch and stores the patch in the patch pool;
        2) randomly selects a patch from the patch pool and pastes it on the
           input (at random position) to simulate occlusion.

    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. arXiv preprint, 2019.
    """

    def __init__(self, p=0.5, pool_capacity=50000, min_sample_size=100,
                 patch_min_area=0.01, patch_max_area=0.5, patch_min_ratio=0.1,
                 prob_rotate=0.5, prob_flip_leftright=0.5, **kwargs):
        self.prob_happen = p

        self.patch_min_area = patch_min_area
        self.patch_max_area = patch_max_area
        self.patch_min_ratio = patch_min_ratio

        self.prob_rotate = prob_rotate
        self.prob_flip_leftright = prob_flip_leftright

        self.patchpool = deque(maxlen=pool_capacity)
        self.min_sample_size = min_sample_size

    def generate_wh(self, W, H):
        area = W * H
        for attempt in range(100):
            target_area = random.uniform(self.patch_min_area, self.patch_max_area) * area
            aspect_ratio = random.uniform(self.patch_min_ratio, 1.0 / self.patch_min_ratio)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < W and h < H:
                return w, h

        return None, None

    def transform_patch(self, patch):
        if random.uniform(0, 1) > self.prob_flip_leftright:
            patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
        if random.uniform(0, 1) > self.prob_rotate:
            patch = patch.rotate(random.randint(-10, 10))
        return patch

    def __call__(self, input_tuple):
        img, mask = input_tuple

        W, H = img.size  # original image size

        # collect new patch
        w, h = self.generate_wh(W, H)
        if w is not None and h is not None:
            x1 = random.randint(0, W - w)
            y1 = random.randint(0, H - h)
            new_patch = img.crop((x1, y1, x1 + w, y1 + h))
            self.patchpool.append(new_patch)

        if len(self.patchpool) < self.min_sample_size:
            return img, mask

        if random.uniform(0, 1) > self.prob_happen:
            return img, mask

        # paste a randomly selected patch on a random position
        patch = random.sample(self.patchpool, 1)[0]
        patchW, patchH = patch.size
        x1 = random.randint(0, W - patchW)
        y1 = random.randint(0, H - patchH)
        patch = self.transform_patch(patch)
        img.paste(patch, (x1, y1))

        return img, mask


class RandomColorJitter(ColorJitter):
    def __init__(self, p=0.5, brightness=0.2, contrast=0.15, saturation=0, hue=0, **kwargs):
        self.p = p
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, input_tuple):
        if random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple
        img = self.forward(img)

        return img, mask


class RandomGrayscale(object):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, input_tuple):
        if random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple

        num_output_channels = 1 if img.mode == 'L' else 3
        img = F.to_grayscale(img, num_output_channels=num_output_channels)

        return img, mask

class Equalize(object):
    def __init__(self, p=0.5, **kwargs):
        self.p = p

    def __call__(self, input_tuple):
        if random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple
        img = ImageOps.equalize(img)

        return img, mask

class Posterize(object):
    def __init__(self, p=0.5, bits=1, **kwargs):
        self.p = p
        self.bits = bits

    def __call__(self, input_tuple):
        if random.uniform(0, 1) > self.p:
            return input_tuple
        bit = random.randint(self.bits, 6)

        img, mask = input_tuple
        img = ImageOps.posterize(img, bit)

        return img, mask

class RandomNegative(object):
    def __init__(self, p=0.1, **kwargs):
        self.p = p

    def __call__(self, input_tuple):
        if random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple
        img = ImageOps.invert(img)

        return img, mask


class ForceGrayscale(object):
    def __init__(self):
        pass

    def __call__(self, input_tuple):
        img, mask = input_tuple

        num_output_channels = 1 if img.mode == 'L' else 3
        img = F.to_grayscale(img, num_output_channels=num_output_channels)

        return img, mask


class RandomPadding(object):
    """Random padding
    """

    def __init__(self, p=0.5, padding=(0, 10), **kwargs):
        self.p = p
        self.padding_limits = padding

    def __call__(self, input_tuple, *args, **kwargs):
        if random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple

        rnd_padding = [random.randint(self.padding_limits[0], self.padding_limits[1]) for _ in range(4)]
        rnd_fill = random.randint(0, 255)

        img = F.pad(img, tuple(rnd_padding), fill=rnd_fill, padding_mode='constant')
        mask = F.pad(mask, tuple(rnd_padding), fill=0, padding_mode='constant') if mask != '' else mask

        return img, mask


class RandomRotate(object):
    """Random rotate
    """

    def __init__(self, p=0.5, angle=(-5, 5), values=None, **kwargs):
        self.p = p
        self.angle = angle

        self.discrete = values is not None and len([v for v in values if v != 0]) > 0
        self.values = values

    def __call__(self, input_tuple, *args, **kwargs):
        if random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple

        if self.discrete:
            rnd_angle = float(self.values[random.randint(0, len(self.values) - 1)])
        else:
            rnd_angle = random.randint(self.angle[0], self.angle[1])

        img = F.rotate(img, rnd_angle, expand=False, center=None)
        if mask != '':
            rgb_mask = mask.convert('RGB')
            rgb_mask = F.rotate(rgb_mask, rnd_angle, expand=False, center=None)
            mask = rgb_mask.convert('L')

        return img, mask

class CoarseDropout(object):
    """CoarseDropout of the rectangular regions in the image.
    Args:
        max_holes (int): Maximum number of regions to zero out.
        max_height (int): Maximum height of the hole.
        max_width (int): Maximum width of the hole.
        min_holes (int): Minimum number of regions to zero out. If `None`,
            `min_holes` is be set to `max_holes`. Default: `None`.
        min_height (int): Minimum height of the hole. Default: None. If `None`,
            `min_height` is set to `max_height`. Default: `None`.
        min_width (int): Minimum width of the hole. If `None`, `min_height` is
            set to `max_width`. Default: `None`.
        fill_value (int, float, lisf of int, list of float): value for dropped pixels.
        mask_fill_value (int, float, lisf of int, list of float): fill value for dropped pixels
            in mask. If None - mask is not affected.
    Targets:
        image, mask
    """

    def __init__(
        self,
        max_holes=8,
        max_height=8,
        max_width=8,
        min_holes=None,
        min_height=None,
        min_width=None,
        fill_value=0,
        mask_fill_value=None,
        p=0.5,
        **kwargs
    ):
        self.p = p
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.min_holes = min_holes if min_holes is not None else max_holes
        self.min_height = min_height if min_height is not None else max_height
        self.min_width = min_width if min_width is not None else max_width
        self.fill_value = fill_value
        self.mask_fill_value = mask_fill_value
        if not 0 < self.min_holes <= self.max_holes:
            raise ValueError("Invalid combination of min_holes and max_holes. Got: {}".format([min_holes, max_holes]))
        if not 0 < self.min_height <= self.max_height:
            raise ValueError(
                "Invalid combination of min_height and max_height. Got: {}".format([min_height, max_height])
            )
        if not 0 < self.min_width <= self.max_width:
            raise ValueError("Invalid combination of min_width and max_width. Got: {}".format([min_width, max_width]))

    def __call__(self, input_tuple):

        if random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple
        height, width = img.size

        holes = []
        for _n in range(random.randint(self.min_holes, self.max_holes)):
            hole_height = random.randint(self.min_height, self.max_height)
            hole_width = random.randint(self.min_width, self.max_width)

            y1 = random.randint(0, height - hole_height)
            x1 = random.randint(0, width - hole_width)
            y2 = y1 + hole_height
            x2 = x1 + hole_width
            holes.append((x1, y1, x2, y2))

        img = self.cutout(img, holes, self.fill_value)
        return img, mask

    @staticmethod
    def cutout(image, holes, fill_value):
        # Make a copy of the input image since we don't want to modify it directly
        image = np.array(image)
        for x1, y1, x2, y2 in holes:
            image[y1:y2, x1:x2] = fill_value

        return Image.fromarray(image)

class RandomGrid(object):
    """Random grid
    """

    def __init__(self, p=0.5, color=-1, grid_size=(24, 64), thickness=(1, 1), angle=(0, 180), **kwargs):
        self.p = p
        self.color = color
        self.grid_size = grid_size
        self.thickness = thickness
        self.angle = angle

    def __call__(self, input_tuple, *args, **kwargs):
        if random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple

        if self.color == (-1, -1, -1):  # Random color
            color = tuple([random.randint(0, 256) for _ in range(3)])
        else:
            color = self.color

        grid_size = random.randint(*self.grid_size)
        thickness = random.randint(*self.thickness)
        angle = random.randint(*self.angle)

        return self.draw_grid(img, grid_size, color, thickness, angle), mask

    @staticmethod
    def draw_grid(image, grid_size, color, thickness, angle):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        mask = np.zeros((h * 8, w * 8, 3), dtype='uint8')
        mask_h, mask_w = mask.shape[:2]
        for i in range(0, mask_h, grid_size):
            p1 = (0, i)
            p2 = (mask_w, i + grid_size)
            mask = cv2.line(mask, p1, p2, (255, 255, 255), thickness)
        for i in range(0, mask_w, grid_size):
            p1 = (i, 0)
            p2 = (i + grid_size, mask_h)
            mask = cv2.line(mask, p1, p2, (255, 255, 255), thickness)

        center = (mask_w // 2, mask_h // 2)

        if angle > 0:
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            mask = cv2.warpAffine(mask, rot_mat, (mask_w, mask_h), flags=cv2.INTER_LINEAR)

        offset = (random.randint(-16, 16), random.randint(16, 16))
        center = (center[0] + offset[0], center[1] + offset[1])
        mask = mask[center[1] - h // 2: center[1] + h // 2, center[0] - w // 2: center[0] + w // 2, :]
        mask = cv2.resize(mask, (w, h))
        assert img.shape == mask.shape
        img = np.where(mask == 0, img, color).astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)


class RandomFigures(object):
    """Insert random figure or some figures from the list [line, rectangle, circle]
    with random color and thickness
    """

    def __init__(self, p=0.5, random_color=True, always_single_figure=False,
                 thicknesses=(1, 6), circle_radiuses=(5, 64), figure_prob=0.5,
                 figures=None, **kwargs):
        self.p = p
        self.random_color = random_color
        self.always_single_figure = always_single_figure
        self.thicknesses = thicknesses
        self.circle_radiuses = circle_radiuses
        self.figure_prob = figure_prob

        if figures is None:
            self.figures = [cv2.line, cv2.rectangle, cv2.circle]
        else:
            assert isinstance(figures, (tuple, list))

            self.figures = []
            for figure in figures:
                assert isinstance(figure, str)

                if hasattr(cv2, figure):
                    self.figures.append(getattr(cv2, figure))
                else:
                    raise ValueError('Unknown figure: {}'.format(figure))

    def __call__(self, input_tuple):
        if random.uniform(0, 1) > self.p:
            return input_tuple

        image, mask = input_tuple

        if self.always_single_figure:
            figure = [self.figures[random.randint(0, len(self.figures) - 1)]]
        else:
            figure = []
            for i in range(len(self.figures)):
                if random.uniform(0, 1) > self.figure_prob:
                    figure.append(self.figures[i])

        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w = cv_image.shape[:2]
        for f in figure:
            p1 = (random.randint(0, w), random.randint(0, h))
            p2 = (random.randint(0, w), random.randint(0, h))
            color = tuple([random.randint(0, 256) for _ in range(3)]) if self.random_color else (0, 0, 0)
            thickness = random.randint(*self.thicknesses)
            if f != cv2.circle:
                cv_image = f(cv_image, p1, p2, color, thickness)
            else:
                r = random.randint(*self.circle_radiuses)
                cv_image = f(cv_image, p1, r, color, thickness)

        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        return Image.fromarray(img), mask


class CutOutWithPrior(object):
    """Cut out around facial landmarks
    """

    def __init__(self, max_area=0.1, p=0.5, **kwargs):
        self.p = p
        self.max_area = max_area

    def __call__(self, input_tuple, *args, **kwargs):
        img, mask = input_tuple

        img = np.array(img)

        height, width = img.shape[:2]
        keypoints_ref = np.zeros((5, 2), dtype=np.float32)
        keypoints_ref[:, 0] = FivePointsAligner.ref_landmarks[:, 0] * width
        keypoints_ref[:, 1] = FivePointsAligner.ref_landmarks[:, 1] * height

        if float(torch.FloatTensor(1).uniform_()) < self.p:
            erase_num = torch.LongTensor(1).random_(1, 4)
            erase_ratio = torch.FloatTensor(1).uniform_(self.max_area / 2, self.max_area)
            erase_h = math.sqrt(erase_ratio) / float(erase_num) * height
            erase_w = math.sqrt(erase_ratio) / float(erase_num) * width

            erased_idx = []
            for _ in range(erase_num):
                erase_pos = int(torch.LongTensor(1).random_(0, 5))
                while erase_pos in erased_idx:
                    erase_pos = int(torch.LongTensor(1).random_(0, 5))

                left_corner = (
                    int(keypoints_ref[erase_pos][0] - erase_h / 2), int(keypoints_ref[erase_pos][1] - erase_w / 2))
                right_corner = (
                    int(keypoints_ref[erase_pos][0] + erase_h / 2), int(keypoints_ref[erase_pos][1] + erase_w / 2))

                cv2.rectangle(img, tuple(left_corner), tuple(right_corner), (0, 0, 0), thickness=-1)
                erased_idx.append(erase_pos)

        img = Image.fromarray(img)
        return img, mask


class GaussianBlur(object):
    """Apply gaussian blur with random parameters
    """

    def __init__(self, p, k):
        self.p = p
        assert k % 2 == 1
        self.k = k

    def __call__(self, input_tuple, *args, **kwargs):
        if random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple

        img = np.array(img)
        img = cv2.blur(img, (self.k, self.k))
        img = Image.fromarray(img)

        return img, mask


class GaussianNoise(object):
    """Adds gaussian noise with random parameters
    """

    def __init__(self, p, sigma, grayscale):
        self.p = p
        self.sigma = sigma
        self.grayscale = grayscale

    def __call__(self, input_tuple, *args, **kwargs):
        if random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple

        image_max_brightness = np.max(img)
        image_min_brightness = np.min(img)
        brightness_range = image_max_brightness - image_min_brightness
        max_noise_sigma = self.sigma * float(brightness_range if brightness_range > 0 else 1)
        noise_sigma = np.random.uniform(0, max_noise_sigma)

        img = np.array(img, dtype=np.float32)

        noise_shape = img.shape[:2] + (1,) if self.grayscale else img.shape
        img += np.random.normal(loc=0.0, scale=noise_sigma, size=noise_shape)

        img[img < 0.0] = 0.0
        img[img > 255.0] = 255.0

        img = Image.fromarray(img.astype(np.uint8))

        return img, mask


class MixUp(object):
    """MixUp augmentation
    """

    def __init__(self, p=0.5, alpha=0.2, images_root_dir=None, images_list_file=None, **kwargs):
        self.p = p
        self.alpha = alpha

        self.enable = images_root_dir is not None and exists(images_root_dir) and \
                      images_list_file is not None and exists(images_list_file)

        if self.enable:
            all_image_files = []
            with open(images_list_file) as input_stream:
                for line in input_stream:
                    image_name = line.replace('\n', '')
                    if len(image_name) > 0:
                        image_path = join(images_root_dir, image_name)
                        all_image_files.append(image_path)
            self.surrogate_image_paths = all_image_files
            assert len(self.surrogate_image_paths) > 0

    def get_num_images(self):
        return len(self.surrogate_image_paths) if self.enable else 0

    def _load_surrogate_image(self, idx, trg_image_size):
        trg_image_width, trg_image_height = trg_image_size

        image = read_image(self.surrogate_image_paths[idx])

        scale = 1.125
        new_width, new_height = int(trg_image_width * scale), int(trg_image_height * scale)
        image = image.resize((new_width, new_height), Image.BILINEAR)

        x_max_range = new_width - trg_image_width
        y_max_range = new_height - trg_image_height
        x1 = int(round(random.uniform(0, x_max_range)))
        y1 = int(round(random.uniform(0, y_max_range)))

        image = image.crop((x1, y1, x1 + trg_image_width, y1 + trg_image_height))

        if random.uniform(0, 1) > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        return image

    def __call__(self, input_tuple, *args, **kwargs):
        if not self.enable or random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple

        alpha = np.random.beta(self.alpha, self.alpha)
        alpha = alpha if alpha < 0.5 else 1.0 - alpha

        surrogate_image_idx = random.randint(0, len(self.surrogate_image_paths) - 1)
        surrogate_image = self._load_surrogate_image(surrogate_image_idx, img.size)

        float_img = np.array(img).astype(np.float32)
        float_surrogate_image = np.array(surrogate_image).astype(np.float32)
        mixed_image = (1.0 - alpha) * float_img + alpha * float_surrogate_image

        out_image = mixed_image.clip(0.0, 255.0).astype(np.uint8)

        return Image.fromarray(out_image), mask


class RandomBackgroundSubstitution(object):
    """Random background substitution augmentation
    """

    def __init__(self, p=0.5, images_root_dir=None, images_list_file=None, **kwargs):
        self.p = p

        self.enable = images_root_dir is not None and exists(images_root_dir) and \
                      images_list_file is not None and exists(images_list_file)

        if self.enable:
            all_image_files = []
            with open(images_list_file) as input_stream:
                for line in input_stream:
                    image_name = line.replace('\n', '')
                    if len(image_name) > 0:
                        image_path = join(images_root_dir, image_name)
                        all_image_files.append(image_path)
            self.surrogate_image_paths = all_image_files
            assert len(self.surrogate_image_paths) > 0

    def get_num_images(self):
        return len(self.surrogate_image_paths) if self.enable else 0

    def _load_bg_image(self, idx, trg_image_size):
        trg_image_width, trg_image_height = trg_image_size

        image = read_image(self.surrogate_image_paths[idx])

        scale = 1.125
        new_width, new_height = int(trg_image_width * scale), int(trg_image_height * scale)
        image = image.resize((new_width, new_height), Image.BILINEAR)

        x_max_range = new_width - trg_image_width
        y_max_range = new_height - trg_image_height
        x1 = int(round(random.uniform(0, x_max_range)))
        y1 = int(round(random.uniform(0, y_max_range)))

        image = image.crop((x1, y1, x1 + trg_image_width, y1 + trg_image_height))

        if random.uniform(0, 1) > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        return image

    def __call__(self, input_tuple, *args, **kwargs):
        if not self.enable or random.uniform(0, 1) > self.p:
            return input_tuple

        img, mask = input_tuple
        if mask == '':
            return input_tuple

        surrogate_image_idx = random.randint(0, len(self.surrogate_image_paths) - 1)
        surrogate_image = self._load_bg_image(surrogate_image_idx, img.size)

        src_img = np.array(img)
        out_img = np.array(surrogate_image)

        fg_mask = np.array(mask).astype(np.bool)
        out_img[fg_mask] = src_img[fg_mask]

        return Image.fromarray(out_img), mask


class DisableBackground(object):
    def __call__(self, input_tuple, *args, **kwargs):
        img, mask = input_tuple
        if mask == '':
            return input_tuple

        bg_mask = ~np.array(mask).astype(np.bool)

        img = np.array(img)
        img[bg_mask] = 0

        return Image.fromarray(img), mask


class PairResize(object):
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        assert isinstance(size, int) or len(size) == 2
        self.size = size
        self.interpolation = interpolation

    def __call__(self, input_tuple):
        image, mask = input_tuple

        image = F.resize(image, self.size, self.interpolation)
        mask = F.resize(mask, self.size, self.interpolation) if mask != '' else mask

        return image, mask


class PairNormalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, input_tuple):
        image, mask = input_tuple

        return F.normalize(image, self.mean, self.std, self.inplace), mask


class PairToTensor(object):
    def __call__(self, input_tuple):
        image, mask = input_tuple

        image = F.to_tensor(image)
        mask = F.to_tensor(mask) if mask != '' else mask

        return image, mask

#FROM_HERE
_FILL = (128, 128, 128)

_MAX_LEVEL = 10.

_HPARAMS_DEFAULT = dict(
    translate_const=250,
    img_mean=_FILL,
)

_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)

class AugmentMixingFabric:
    def __init__(self, image_mean, translate_const) -> None:
        self._max_level = 10.
        self._random_interpolation = (Image.BILINEAR, Image.BICUBIC)
        self._hparams_default = dict(
            translate_const=translate_const,
            img_mean=image_mean,
        )

        self._random_interpolation = (Image.BILINEAR, Image.BICUBIC)

        self.LEVEL_TO_ARG = {
        'AutoContrast': None,
        'Equalize': None,
        'Rotate': _rotate_level_to_arg,
        'PosterizeIncreasing': _posterize_increasing_level_to_arg,
        'SolarizeIncreasing': _solarize_increasing_level_to_arg,
        'ColorIncreasing': _enhance_increasing_level_to_arg,
        'ContrastIncreasing': _enhance_increasing_level_to_arg,
        'BrightnessIncreasing': _enhance_increasing_level_to_arg,
        'SharpnessIncreasing': _enhance_increasing_level_to_arg,
        'ShearX': _shear_level_to_arg,
        'ShearY': _shear_level_to_arg,
        'TranslateXRel': _translate_rel_level_to_arg,
        'TranslateYRel': _translate_rel_level_to_arg,
    }

    NAME_TO_OP = {
        'AutoContrast': auto_contrast,
        'Equalize': equalize,
        'Rotate': rotate,
        'PosterizeIncreasing': posterize,
        'SolarizeIncreasing': solarize,
        'ColorIncreasing': color,
        'ContrastIncreasing': contrast,
        'BrightnessIncreasing': brightness,
        'SharpnessIncreasing': sharpness,
        'ShearX': self.shear_x,
        'ShearY': self.shear_y,
        'TranslateXRel': translate_x_rel,
        'TranslateYRel': translate_y_rel,
    }

    _AUGMIX_TRANSFORMS = [
        'AutoContrast',
        'ColorIncreasing',  # not in paper
        'ContrastIncreasing',  # not in paper
        'BrightnessIncreasing',  # not in paper
        'SharpnessIncreasing',  # not in paper
        'Equalize',
        'Rotate',
        'PosterizeIncreasing',
        'SolarizeIncreasing',
        'ShearX',
        'ShearY',
        'TranslateXRel',
        'TranslateYRel',
    ]


    @staticmethod
    def _interpolation(kwargs):
        interpolation = kwargs.pop('resample', Image.BILINEAR)
        if isinstance(interpolation, (list, tuple)):
            return random.choice(interpolation)
        else:
            return interpolation

    @staticmethod
    def _check_args_tf(kwargs):
        kwargs['resample'] = self._interpolation(kwargs)

    @staticmethod
    def shear_x(img, factor, **kwargs):
        _check_args_tf(kwargs)
        return img.transform(img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs)

    @staticmethod
    def shear_y(img, factor, **kwargs):
        _check_args_tf(kwargs)
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), **kwargs)

    @staticmethod
    def translate_x_rel(img, pct, **kwargs):
        pixels = pct * img.size[0]
        _check_args_tf(kwargs)
        return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)

    @staticmethod
    def translate_y_rel(img, pct, **kwargs):
        pixels = pct * img.size[1]
        _check_args_tf(kwargs)
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)

    @staticmethod
    def rotate(img, degrees, **kwargs):
        _check_args_tf(kwargs)
        return img.rotate(degrees, **kwargs)

    @staticmethod
    def auto_contrast(img, **__):
        return ImageOps.autocontrast(img)

    @staticmethod
    def equalize(img, **__):
        return ImageOps.equalize(img)

    @staticmethod
    def solarize(img, thresh, **__):
        return ImageOps.solarize(img, thresh)

    @staticmethod
    def posterize(img, bits_to_keep, **__):
        if bits_to_keep >= 8:
            return img
        return ImageOps.posterize(img, bits_to_keep)

    @staticmethod
    def contrast(img, factor, **__):
        return ImageEnhance.Contrast(img).enhance(factor)

    @staticmethod
    def color(img, factor, **__):
        return ImageEnhance.Color(img).enhance(factor)

    @staticmethod
    def brightness(img, factor, **__):
        return ImageEnhance.Brightness(img).enhance(factor)

    @staticmethod
    def sharpness(img, factor, **__):
        return ImageEnhance.Sharpness(img).enhance(factor)

    @staticmethod
    def _randomly_negate(v):
        """With 50% prob, negate the value"""
        return -v if random.random() > 0.5 else v

    @staticmethod
    def _rotate_level_to_arg(level, _hparams):
        # range [-30, 30]
        level = (level / _MAX_LEVEL) * 30.
        level = _randomly_negate(level)
        return level,

    @staticmethod
    def _enhance_increasing_level_to_arg(level, _hparams):
        # the 'no change' level is 1.0, moving away from that towards 0. or 2.0 increases the enhancement blend
        # range [0.1, 1.9]
        level = (level / _MAX_LEVEL) * .9
        level = 1.0 + _randomly_negate(level)
        return level,

    @staticmethod
    def _shear_level_to_arg(level, _hparams):
        # range [-0.3, 0.3]
        level = (level / _MAX_LEVEL) * 0.3
        level = _randomly_negate(level)
        return level,

    @staticmethod
    def _translate_rel_level_to_arg(level, hparams):
        # default range [-0.45, 0.45]
        translate_pct = hparams.get('translate_pct', 0.45)
        level = (level / _MAX_LEVEL) * translate_pct
        level = _randomly_negate(level)
        return level,

    @staticmethod
    def _posterize_level_to_arg(level, _hparams):
        # As per Tensorflow TPU EfficientNet impl
        # range [0, 4], 'keep 0 up to 4 MSB of original image'
        # intensity/severity of augmentation decreases with level
        return int((level / _MAX_LEVEL) * 4),

    @staticmethod
    def _posterize_increasing_level_to_arg(level, hparams):
        # As per Tensorflow models research and UDA impl
        # range [4, 0], 'keep 4 down to 0 MSB of original image',
        # intensity/severity of augmentation increases with level
        return 4 - _posterize_level_to_arg(level, hparams)[0],

    @staticmethod
    def _solarize_level_to_arg(level, _hparams):
        # range [0, 256]
        # intensity/severity of augmentation decreases with level
        return int((level / _MAX_LEVEL) * 256),

    @staticmethod
    def _solarize_increasing_level_to_arg(level, _hparams):
        # range [0, 256]
        # intensity/severity of augmentation increases with level
        return 256 - _solarize_level_to_arg(level, _hparams)[0],


class AugmentOp:
    def __init__(self, name, prob=0.5, magnitude=10, hparams=None):
        hparams = hparams
        self.aug_fn = NAME_TO_OP[name]
        self.level_fn = LEVEL_TO_ARG[name]
        self.prob = prob
        self.magnitude = magnitude
        self.hparams = hparams.copy()
        self.kwargs = dict(
            fillcolor=hparams['img_mean'],
            resample=hparams['interpolation'],
        )

        # If magnitude_std is > 0, we introduce some randomness
        # in the usually fixed policy and sample magnitude from a normal distribution
        # with mean `magnitude` and std-dev of `magnitude_std`.
        # NOTE This is my own hack, being tested, not in papers or reference impls.
        # If magnitude_std is inf, we sample magnitude from a uniform distribution
        self.magnitude_std = self.hparams.get('magnitude_std', 0)

    def __call__(self, img):
        if self.prob < 1.0 and random.random() > self.prob:
            return img
        magnitude = self.magnitude
        if self.magnitude_std:
            if self.magnitude_std == float('inf'):
                magnitude = random.uniform(0, magnitude)
            elif self.magnitude_std > 0:
                magnitude = random.gauss(magnitude, self.magnitude_std)
        magnitude = min(_MAX_LEVEL, max(0, magnitude))  # clip to valid range
        level_args = self.level_fn(magnitude, self.hparams) if self.level_fn is not None else tuple()
        return self.aug_fn(img, *level_args, **self.kwargs)

def augmix_ops(magnitude=10, hparams=None, transforms=None):
    hparams = hparams or _hparams_default
    transforms = transforms or _AUGMIX_TRANSFORMS
    return [AugmentOp(
        name, prob=1.0, magnitude=magnitude, hparams=hparams) for name in transforms]


class AugMixAugment:
    """ AugMix Transform
    Adapted and improved from impl here: https://github.com/google-research/augmix/blob/master/imagenet.py
    From paper: 'AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty -
    https://arxiv.org/abs/1912.02781
    """
    def __init__(self, ops, alpha=1., width=3, depth=-1, blended=False, p=1.):
        self.ops = ops
        self.alpha = alpha
        self.width = width
        self.depth = depth
        self.blended = blended  # blended mode is faster but not well tested
        self.p = 1.

    def _calc_blended_weights(self, ws, m):
        ws = ws * m
        cump = 1.
        rws = []
        for w in ws[::-1]:
            alpha = w / cump
            cump *= (1 - alpha)
            rws.append(alpha)
        return np.array(rws[::-1], dtype=np.float32)

    def _apply_blended(self, img, mixing_weights, m):
        # This is my first crack and implementing a slightly faster mixed augmentation. Instead
        # of accumulating the mix for each chain in a Numpy array and then blending with original,
        # it recomputes the blending coefficients and applies one PIL image blend per chain.
        # TODO the results appear in the right ballpark but they differ by more than rounding.
        img_orig = img.copy()
        ws = self._calc_blended_weights(mixing_weights, m)
        for w in ws:
            depth = self.depth if self.depth > 0 else np.random.randint(1, 4)
            ops = np.random.choice(self.ops, depth, replace=True)
            img_aug = img_orig  # no ops are in-place, deep copy not necessary
            for op in ops:
                img_aug = op(img_aug)
            img = Image.blend(img, img_aug, w)
        return img

    def _apply_basic(self, img, mixing_weights, m):
        # This is a literal adaptation of the paper/official implementation without normalizations and
        # PIL <-> Numpy conversions between every op. It is still quite CPU compute heavy compared to the
        # typical augmentation transforms, could use a GPU / Kornia implementation.
        img_shape = img.size[0], img.size[1], len(img.getbands())
        mixed = np.zeros(img_shape, dtype=np.float32)
        for mw in mixing_weights:
            depth = self.depth if self.depth > 0 else np.random.randint(1, 4)
            ops = np.random.choice(self.ops, depth, replace=True)
            img_aug = img  # no ops are in-place, deep copy not necessary
            for op in ops:
                img_aug = op(img_aug)
            mixed += mw * np.asarray(img_aug, dtype=np.float32)
        np.clip(mixed, 0, 255., out=mixed)
        mixed = Image.fromarray(mixed.astype(np.uint8))
        return Image.blend(img, mixed, m)

    def __call__(self, input_tuple):
        img, mask = input_tuple
        mixing_weights = np.float32(np.random.dirichlet([self.alpha] * self.width))
        m = np.float32(np.random.beta(self.alpha, self.alpha))
        if self.blended:
            mixed = self._apply_blended(img, mixing_weights, m)
        else:
            mixed = self._apply_basic(img, mixing_weights, m)
        return mixed, mask

def augment_and_mix_transform(config_str, hparams=_hparams_default):
    """ Create AugMix PyTorch transform
    :param config_str: String defining configuration of random augmentation. Consists of multiple sections separated by
    dashes ('-'). The first section defines the specific variant of rand augment (currently only 'rand'). The remaining
    sections, not order sepecific determine
        'm' - integer magnitude (severity) of augmentation mix (default: 3)
        'w' - integer width of augmentation chain (default: 3)
        'd' - integer depth of augmentation chain (-1 is random [1, 3], default: -1)
        'b' - integer (bool), blend each branch of chain into end result without a final blend, less CPU (default: 0)
        'mstd' -  float std deviation of magnitude noise applied (default: 0)
    Ex 'augmix-m5-w4-d2' results in AugMix with severity 5, chain width 4, chain depth 2
    :param hparams: Other hparams (kwargs) for the Augmentation transforms
    :return: A PyTorch compatible Transform
    """
    magnitude = 3
    width = 3
    depth = -1
    alpha = 1.
    blended = False
    hparams['magnitude_std'] = float('inf')
    config = config_str.split('-')
    assert config[0] == 'augmix'
    config = config[1:]
    for c in config:
        cs = re.split(r'(\d.*)', c)
        if len(cs) < 2:
            continue
        key, val = cs[:2]
        if key == 'mstd':
            # noise param injected via hparams for now
            hparams.setdefault('magnitude_std', float(val))
        elif key == 'm':
            magnitude = int(val)
        elif key == 'w':
            width = int(val)
        elif key == 'd':
            depth = int(val)
        elif key == 'a':
            alpha = float(val)
        elif key == 'b':
            blended = bool(val)
        else:
            assert False, 'Unknown AugMix config section'
    ops = augmix_ops(magnitude=magnitude, hparams=hparams)
    return AugMixAugment(ops, alpha=alpha, width=width, depth=depth, blended=blended)

# class AugMix(object):
#     def __init__(self, transforms: list, width=3, depth=-1, alpha=1., p=1., **kwargs):
#         self.transforms = transforms
#         self.width = width
#         self.depth = depth
#         self.alpha = alpha
#         self.p = p
#         self.random_flip = None
#         self.random_rotate = None
#         self.random_crop = None
#         self.center_crop = None
#         self.coarse_dropout = None
#         self.train_transforms = []
#         for aug in self.transforms:
#             if aug.__class__.__name__ == 'PairResize':
#                 self.resize = aug
#             if aug.__class__.__name__ == 'RandomRotate':
#                 self.random_rotate = aug
#             elif aug.__class__.__name__ == 'RandomHorizontalFlip':
#                 self.random_flip = aug
#             elif aug.__class__.__name__ == 'RandomCrop':
#                 self.random_crop = aug
#             elif aug.__class__.__name__ == 'CenterCrop':
#                 self.center_crop = aug
#             elif aug.__class__.__name__ == 'CoarseDropout':
#                 self.coarse_dropout = aug
#             else:
#                 self.train_transforms.append(aug)

#     def __call__(self, input_tuple):
#         # do augmentation before resize
#         for augment in [self.random_crop, self.center_crop]:
#             if augment != None:
#                 input_tuple = augment(input_tuple)
#         # resize image always
#         input_tuple = self.resize(input_tuple)
#         # do flip augmentation if it's avaible
#         input_tuple = self.random_flip(input_tuple) if self.random_flip else input_tuple
#         input_tuple = self.coarse_dropout(input_tuple) if self.coarse_dropout else input_tuple
#         input_tuple = self.random_rotate(input_tuple) if self.random_rotate else input_tuple
#         image, mask = input_tuple
#         if self.train_transforms:
#             # run augmix pipeline
#             ws = np.float32(np.random.dirichlet([self.alpha] * self.width))
#             m = np.float32(np.random.beta(self.alpha, self.alpha))

#             mix_img = np.zeros_like(image)
#             mix_mask = np.zeros_like(mask) if mask != '' else mask
#             for i in range(self.width):
#                 image_aug = image.copy()
#                 mask_aug = mask.copy() if mask != '' else mask
#                 depth = self.depth if self.depth > 0 else np.random.randint(1, 4)

#                 for _ in range(depth):
#                     op = np.random.choice(self.train_transforms)
#                     image_aug, mask_aug = op((image_aug, mask_aug))

#                 mix_img = mix_img + ws[i] * image_aug
#                 mix_mask = mix_mask + ws[i] * mask_aug if mask != '' else mask
#             prob = np.random.rand(1)[0]
#             if prob <= self.p:
#                 mixed_image = (1 - m) * image + m * mix_img
#                 mixed_mask = (1 - m) * mask + m * mix_mask if mask != '' else mask
#                 # # converting to PIL
#                 mixed_mask = Image.fromarray(mixed_mask.astype(np.uint8)) if mixed_mask != '' else mixed_mask
#                 return Image.fromarray(mixed_image.astype(np.uint8)), mixed_mask

#         assert not self.transforms or prob > self.p
#         return image, mask

def build_transforms(height, width, transforms=None, norm_mean=(0.485, 0.456, 0.406),
                     norm_std=(0.229, 0.224, 0.225), apply_masks_to_test=False, **kwargs):
    """Builds train and test transform functions.

    Args:
        height (int): target image height.
        width (int): target image width.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): normalization mean values. Default is ImageNet means.
        norm_std (list or None, optional): normalization standard deviation values. Default is
            ImageNet standard deviation values.
    """
    if transforms is None:
        return None, None

    if norm_mean is None or norm_std is None:
        norm_mean = [0.485, 0.456, 0.406]  # imagenet mean
        norm_std = [0.229, 0.224, 0.225]  # imagenet std

    print('Building train transforms ...')
    transform_tr = []
    if transforms.random_grid.enable and transforms.random_grid.before_resize:
        print('+ random_grid')
        transform_tr += [RandomGrid(**transforms.random_grid)]
    if transforms.random_figures.enable and transforms.random_figures.before_resize:
        print('+ random_figures')
        transform_tr += [RandomFigures(**transforms.random_figures)]
    if transforms.center_crop.enable and not transforms.center_crop.test_only:
        print('+ center_crop')
        transform_tr += [CenterCrop(margin=transforms.center_crop.margin)]
    if transforms.random_padding.enable:
        print('+ random_padding')
        transform_tr += [RandomPadding(**transforms.random_padding)]
    if transforms.random_crop.enable:
        print('+ random crop')
        transform_tr += [RandomCrop(p=transforms.random_crop.p,
                                    scale=transforms.random_crop.scale,
                                    margin=transforms.random_crop.margin,
                                    align_ar=transforms.random_crop.scale,
                                    align_center=transforms.random_crop.align_center,
                                    target_ar=float(height)/float(width))]
    print('+ resize to {}x{}'.format(height, width))
    transform_tr += [PairResize((height, width))]
    if transforms.random_background_substitution.enable:
        aug_module = RandomBackgroundSubstitution(**transforms.random_background_substitution)
        if aug_module.enable:
            print('+ random_background_substitution')
            transform_tr += [aug_module]
    if transforms.random_grid.enable and not transforms.random_grid.before_resize:
        print('+ random_grid')
        transform_tr += [RandomGrid(**transforms.random_grid)]
    if transforms.random_figures.enable and not transforms.random_figures.before_resize:
        print('+ random_figures')
        transform_tr += [RandomFigures(**transforms.random_figures)]
    if transforms.random_flip.enable:
        print('+ random flip')
        transform_tr += [RandomHorizontalFlip(p=transforms.random_flip.p)]
    if transforms.cut_out_with_prior.enable:
        print('+ cut out with prior')
        transform_tr += [CutOutWithPrior(p=transforms.cut_out_with_prior.p,
                                         max_area=transforms.cut_out_with_prior.max_area)]
    if transforms.random_blur.enable:
        print('+ random_blur')
        transform_tr += [GaussianBlur(p=transforms.random_blur.p,
                                      k=transforms.random_blur.k)]
    if transforms.random_noise.enable:
        print('+ random_noise')
        transform_tr += [GaussianNoise(p=transforms.random_noise.p,
                                       sigma=transforms.random_noise.sigma,
                                       grayscale=transforms.random_noise.grayscale)]
    if transforms.mixup.enable:
        mixup_augmentor = MixUp(**transforms.mixup)
        print('+ mixup (with {} extra images)'.format(mixup_augmentor.get_num_images()))
        transform_tr += [mixup_augmentor]
    if transforms.random_patch.enable:
        print('+ random patch')
        transform_tr += [RandomPatch(**transforms.random_patch)]
    if transforms.color_jitter.enable:
        print('+ color jitter')
        transform_tr += [RandomColorJitter(**transforms.color_jitter)]
    if transforms.random_gray_scale.enable:
        print('+ random_gray_scale')
        transform_tr += [RandomGrayscale(p=transforms.random_gray_scale.p)]
    if transforms.random_rotate.enable:
        print('+ random_rotate')
        transform_tr += [RandomRotate(**transforms.random_rotate)]
    if transforms.equalize.enable:
        print('+ equalize')
        transform_tr += [Equalize(**transforms.equalize)]
    if transforms.posterize.enable:
        print('+ posterize')
        transform_tr += [Posterize(**transforms.posterize)]
    if transforms.random_erase.enable and not transforms.random_erase.norm_image:
        print('+ random erase')
        transform_tr += [RandomErasing(**transforms.random_erase)]
    if transforms.random_negative.enable:
        print('+ random negative')
        transform_tr += [RandomNegative(**transforms.random_negative)]
    if transforms.force_gray_scale.enable:
        print('+ force_gray_scale')
        transform_tr += [ForceGrayscale()]
    if transforms.coarse_dropout.enable:
        print('+ coarse_dropout')
        transform_tr += [CoarseDropout(**transforms.coarse_dropout)]
    if transforms.augmix.enable:
        print('+ AugMix')
        transform_tr += [augment_and_mix_transform(transforms.augmix.cfg_str)]

    print('+ to torch tensor of range [0, 1]')
    transform_tr += [PairToTensor()]
    print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))
    transform_tr += [PairNormalize(mean=norm_mean, std=norm_std)]
    if transforms.random_erase.enable and transforms.random_erase.norm_image:
        print('+ random erase')
        transform_tr += [RandomErasing(**transforms.random_erase)]

    transform_tr = Compose(transform_tr)
    transform_te = build_test_transform(height, width, norm_mean, norm_std, apply_masks_to_test, transforms)
    return transform_tr, transform_te


def build_test_transform(height, width, norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225),
                         apply_masks_to_test=False, transforms=None, **kwargs):
    def get_resize(h, w, scale):
        t_h, t_w = int(h * scale), int(w * scale)
        print('+ resize to {}x{}'.format(t_h, t_w))
        return PairResize((t_h, t_w))
    print('Building test transforms ...')
    transform_te = []
    if transforms.test.resize_first:
        transform_te.append(get_resize(height, width, transforms.test.resize_scale))
    if transforms.center_crop.enable:
        print('+ center_crop')
        transform_te.append(CenterCrop(margin=transforms.center_crop.margin))
    if apply_masks_to_test:
        print('+ background zeroing')
        transform_te.append(DisableBackground())
    if not transforms.test.resize_first:
        transform_te.append(get_resize(height, width, transforms.test.resize_scale))
    if transforms is not None and transforms.force_gray_scale.enable:
        print('+ force_gray_scale')
        transform_te.append(ForceGrayscale())
    print('+ to torch tensor of range [0, 1]')
    transform_te.append(PairToTensor())
    print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))
    transform_te.append(PairNormalize(mean=norm_mean, std=norm_std))
    transform_te = Compose(transform_te)

    return transform_te


def build_inference_transform(height, width, norm_mean=(0.485, 0.456, 0.406),
                              norm_std=(0.229, 0.224, 0.225), **kwargs):
    print('Building inference transforms ...')
    print('+ resize to {}x{}'.format(height, width))
    print('+ to torch tensor of range [0, 1]')
    print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))
    transform_te = Compose([
        Resize((height, width)),
        ToTensor(),
        Normalize(mean=norm_mean, std=norm_std),
    ])

    return transform_te
