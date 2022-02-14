import numpy as np
import numba
import cv2
import os
from pathlib import Path
from tqdm import tqdm
import os.path as osp
import shutil
import pickle
from PIL import Image
from utils.data_utils import _numpy_to_tensor, _load_cpu

@numba.jit()
def distortion(x):
    marginx1 = np.random.rand() * 0.16 - 0.08
    marginy1 = np.random.rand() * 0.16 - 0.08
    marginx2 = np.random.rand() * 0.16 - 0.08
    marginy2 = np.random.rand() * 0.16 - 0.08
    height = len(x)
    width = len(x[0])
    out = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            u = i + height * np.sin(j * marginx2 + i * j * marginy2 / height) * marginx1
            v = j + width * np.sin(i * marginy2 + i * j * marginx1 / width) * marginy1
            u = max(min(height - 1, u), 0)
            v = max(min(width - 1, v), 0)
            uu = int(u)
            vv = int(v)
            out[i, j] = x[uu, vv]
    return out

def randomMaskErase(x, max_num=4, s_l=0.02, s_h=0.3, r_1=0.3, r_2=1 / 0.3, v_l=10.0, v_h=10.0):
    [img_h, img_w, img_c] = x.shape
    out = x.copy()
    num = int(np.sqrt(np.random.randint(1, max_num * max_num)))

    for i in range(num):
        s = np.random.uniform(s_l, s_h) * img_h * img_w
        r = np.random.uniform(r_1, r_2)
        w = int(np.sqrt(s / r))
        h = int(np.sqrt(s * r))
        left = np.random.randint(0, img_w)
        top = np.random.randint(0, img_h)

        mask = np.zeros((img_h, img_w))
        mask[top:min(top + h, img_h), left:min(left + w, img_w)] = 1
        mask = distortion(mask)
        c0 = np.random.uniform(v_l, v_h)
        c1 = np.random.uniform(v_l, v_h)
        c2 = np.random.uniform(v_l, v_h)
        out0 = out[:, :, 0]
        out0[mask > 0] = c0
        out1 = out[:, :, 1]
        out1[mask > 0] = c1
        out2 = out[:, :, 2]
        out2[mask > 0] = c2
    return

def randomErase(img, sl=0.1, sh=0.1,r1=0.3, r2=0.3):
    img = np.array(img)
    while True:
        img_h, img_w, img_c = img.shape
        img_area = img_h * img_w
        mask_area = np.random.uniform(sl,sh) * img_area
        mask_aspect_ratio = np.random.uniform(r1,r2)
        mask_w = int(np.sqrt(mask_area / mask_aspect_ratio))
        mask_h = int(np.sqrt(mask_area * mask_aspect_ratio))
        mask = np.random.rand(mask_h, mask_w, img_c) * 1
        left = np.random.randint(0, img_w)
        top = np.random.randint(0, img_h)
        right = left+mask_w
        bottom = top + mask_h
        if right <= img_w and bottom <= img_h:
            break
    img[top:bottom, left:right,:] = mask
    return Image.fromarray(img)


def main():
    root = '00W-LP-Aug/train_aug_120x120'
    dst = '3D-Dataset/aug_occlu_300wlp'
    filelist_train = '../train_configs/train_aug_120x120.list.train'
    filelist_val = '../train_configs/train_aug_120x120.list.val'
    param_train = '../train_configs/param_all_norm.pkl'
    param_val = '../train_configs/param_all_norm_val.pkl'

    filenames = Path(filelist_val).read_text().strip().split('\n')
    params = _load_cpu(param_val)

    txt_filename = open('../results1/train_aug_120x120_val_aug.txt', 'a')
    pklfile_params = open('../results1/param_all_norm_val_aug.pkl', 'wb')

    new_params = []

    with tqdm(total=len(filenames)) as tqm: # len(filenames)
        for index in range(len(filenames)):

            txt_filename.write(filenames[index])
            txt_filename.write('\n')
            new_params.append(params[index])

            ori_img = Image.open(osp.join(root,filenames[index]))
            new_img = randomErase(ori_img)

            save_path = osp.join(dst,filenames[index][:-4]+'_1'+'.jpg')
            new_img.save(save_path)

            txt_filename.write(filenames[index][:-4]+'_1'+'.jpg')
            txt_filename.write('\n')
            new_params.append(params[index])

            tqm.set_description('Processing %i' % (index))
            tqm.update()

        txt_filename.close()
    tqm.close()
    results = np.array(new_params, dtype=np.float32).reshape((len(params)*2, 62)) # len(params)*
    pickle.dump(results, pklfile_params)
    pklfile_params.close()

    return 0

def fusion_dataset():
    folder1 = '300W-LP-Aug/train_aug_120x120'
    folder2 = '3D-Dataset/aug_occlu_300wlp'
    dst = '3D-Dataset/300W-LP-Aug-Occlusion'

    filenames = os.listdir(folder2)
    with tqdm(total=len(filenames)) as tqm:
        for filename in filenames:
            ori_path = osp.join(folder2, filename)
            new_path = osp.join(dst, filename)
            shutil.copyfile(ori_path, new_path)

            tqm.set_description('Processing')
            tqm.update()

    tqm.close()
    return 0


if __name__ == '__main__':
    main()
    fusion_dataset()
