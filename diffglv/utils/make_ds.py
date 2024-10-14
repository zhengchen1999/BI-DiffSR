import argparse
from PIL import Image
import os
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm
from basicsr.utils import scandir


def make_downsampling(input_folder, save_folder, scale, rescaling=False, downsample_type='bicubic',
                      n_thread=20, wash_only=False):
    """Crop images to subimages.
    Args:
        opt (dict): Configuration dict. It contains:
        input_folder (str): Path to the input folder.
        save_folder (str): Path to save folder.
        n_thread (int): Thread number.
    """
    opt = {}
    opt['scale'] = scale
    opt['rescaling'] = rescaling
    opt['save_folder'] = save_folder
    opt['downsample_type'] = downsample_type
    opt['wash_only'] = wash_only

    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')
    else:
        print(f'Folder {save_folder} already exists. Exit.')
        sys.exit(1)

    img_list = list(scandir(input_folder, full_path=True))

    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(n_thread)
    for path in img_list:
        pool.apply_async(worker, args=(path, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')


def worker(path, opt):
    """Worker for each process.
    Args:
        path (str): Image path.
        opt (dict): Configuration dict. It contains:
        crop_size (int): Crop size.
        step (int): Step for overlapped sliding window.
        thresh_size (int): Threshold size. Patches whose size is lower than thresh_size will be dropped.
        save_folder (str): Path to save folder.
        compression_level (int): for cv2.IMWRITE_PNG_COMPRESSION.
    Returns:
        process_info (str): Process information displayed in progress bar.
    """
    scale = opt['scale']
    save_folder = opt['save_folder']
    downsample_type = opt['downsample_type']

    hr = Image.open(path)
    w, h = hr.size
    if w % scale + h % scale:
        print('\n\nHR needs data washing\n')
        hr = hr.crop([0, 0, w //scale * scale, h //scale *scale])

    ds_func = Image.Resampling.BICUBIC if downsample_type == 'bicubic' else Image.Resampling.BILINEAR

    if not opt['wash_only']:
        lr = hr.resize((w//scale, h//scale), ds_func)
        if opt['rescaling']:
            lr = lr.resize((w//scale*scale, h//scale*scale), ds_func)
    else:
        lr = hr

    lr.save(osp.join(save_folder, osp.split(path)[-1]))
    process_info = f'Processing {osp.split(path)[-1]} ...'
    return process_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str)
    parser.add_argument('--dst', type=str)
    parser.add_argument('--scale', type=int)
    parser.add_argument('--rescaling', '-rs', action='store_true')
    parser.add_argument('--n_worker', type=int, default=20)
    parser.add_argument('--ds_func', type=str, default='bicubic')
    parser.add_argument('--wash_only', '-wo', action='store_true')

    args = parser.parse_args()

    make_downsampling(args.src, args.dst, args.scale, args.rescaling, args.ds_func, args.n_worker, args.wash_only)