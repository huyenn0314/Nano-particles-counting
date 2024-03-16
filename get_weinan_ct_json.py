import numpy as np
import scipy.io
from PIL import Image
import os
import pdb
from scipy.ndimage import label, generate_binary_structure
import json
import argparse
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from skimage import color, morphology
from skimage import util 
import cv2
from scipy.signal.signaltools import wiener


LOG_DIR = "/home/huyentn2/huyen/project/Weinan_prediction/"




def preprocess(img_dir, noise_d, se1, se2):
    img = Image.open(img_dir)
    # plt.imshow(img, cmap='gray')
    inverted_img = util.invert(np.array(img).astype(float))

    # pdb.set_trace()

    # plt.imshow(inverted_img, cmap='gray')
    footprint = morphology.disk(se1)
    res = morphology.white_tophat(inverted_img, footprint)
    # I_bgr = np.array(img) - res
    I_bgr = util.invert(res).astype(float)

    spectrum = np.fft.fftshift(np.fft.fft2(I_bgr))

    mask = np.ones((spectrum.shape))
    m1, n1 = spectrum.shape[0], spectrum.shape[1]
    if n1 %2 == 0:
        mask[:, int(n1/2)] = 0
    elif n1 %2 == 1:
        mask[:, int((n1-1)/2)+1] = 0

    img_back=np.fft.ifft2(np.multiply(mask,spectrum))

    normalized = 255 - abs(img_back)

    A = np.double(normalized)
    out = np.zeros(A.shape, np.double)
    normalized = cv2.normalize(A, out, 1.0, 0.0, cv2.NORM_MINMAX)


    # noise_d = 6
    filtered_img = wiener(normalized, (noise_d, noise_d))

    # se2 = 2
    dilated_img = cv2.dilate(util.invert(filtered_img), np.ones((se2,se2), np.uint8))
    dilated_img = util.invert(dilated_img)
    # plt.imshow(dilated_img, cmap='gray')
    
    return dilated_img



def _visualize_preshape(img, dmap, img_p):
    """Draw a density map onto the image with the same shape as input array"""
    # keep the same aspect ratio as an input image

    fig, ax = plt.subplots(figsize=figaspect(1.0 * img.shape[1] / img.shape[0]))
    fig.subplots_adjust(0, 0, 1, 1)

    # plot a density map without axis
    # ax.imshow(dmap, cmap="hot")
    plt.axis('off')

    # ax.imshow(color.label2rgb(dmap/255, img/255, colors=[(255,0,0),(0,0,255)],alpha=0.001, bg_label=0, bg_color=None))
    gh = color.label2rgb(dmap/255, img/255, colors=[(255,0,0),(0,0,255)],alpha=0.003, bg_label=0, bg_color=None)
    gh = 255*(gh-np.min(gh)) / (np.max(gh)-np.min(gh))
    # plt.imsave("THYRTT.png", gh.astype("uint8"))

    # plt.savefig("THYRTT.png", bbox_inches='tight', pad_inches=0, dpi=100)
    # pdb.set_trace()  
    # plt.savefig(img_p, bbox_inches='tight', pad_inches=0, dpi=1)
    # 
    plt.imsave(os.path.join("/home/huyentn2/huyen/project/Weinan_prediction/vis/wein_heatmap", os.path.basename(img_p)), gh.astype("uint8"))

    # plt.imsave("THYRTT.png", img.astype("uint8"), cmap ='gray')

    plt.imsave(os.path.join("/home/huyentn2/huyen/project/Weinan_prediction/vis/preprocessed_imgs", os.path.basename(img_p)), img.astype("uint8"), cmap = "gray")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type = int, dest='fold', default= 1)   
    parser.add_argument('--d', type = str, dest='dataset_name', default= 'particle')     
    parser.add_argument('--noise_d', type = int, dest='noise_d', default= 6)
    parser.add_argument('--se1', type = int, dest='se1', default= 5)    # the radius of rings used for tophat filter
    parser.add_argument('--se2', type = int, dest='se2', default= 2)   


    args = parser.parse_args()
    # mat = scipy.io.loadmat('/home/huyentn2/huyen/project/Weinan_prediction/cell_count.mat')

    # print(mat)
    log_path = os.path.join(LOG_DIR, "out_wein/{}/{}".format(args.dataset_name, args.fold))

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    k= args.fold


    f = open("/home/huyentn2/huyen/copy_cig/cell_counting/objects_counting_dmap_JUNO/particle/keep_index/test_val_train_set_{}.json".format(k))
    fold_files = json.load(f)['test_images']

    fold_files = [os.path.basename(img) for img in fold_files]

    # pdb.set_trace()
 
    pred_wein = {}
    dir = "/home/huyentn2/huyen/project/Weinan_prediction/output_v20_png/"
    img_dir = "/home/huyentn2/huyen/copy_cig/datatset/save_patch/"
    # files = os.listdir(dir)
    for file in fold_files:
        img = np.array(Image.open(dir + file))

        # pdb.set_trace()

        labeled_array, num_features = label(img)

        pred_wein[file] = num_features

        # pdb.set_trace()
        
        # im = np.array(Image.open(img_dir + file))
        image_equalized = preprocess(img_dir + file, args.noise_d, args.se1, args.se2)

        image_equalized = (image_equalized - np.min(image_equalized)) / (np.max(image_equalized) - np.min(image_equalized))
        preprocessed_img = (image_equalized*255).astype(np.uint8)
               
        _visualize_preshape(preprocessed_img, img, file)



    with open(log_path + "/wein_count_fold{}.json".format(k), "w") as outfile:
        json.dump(pred_wein, outfile)


    


#  python huyen/project/Weinan_prediction/test.py



# (cv549) huyentn2@cig-01:~/huyen/project/density_est/data$ ls  where all data is
# (cv549) huyentn2@cig-01:~/huyen/project/data_nano/han$ ls
# meth1  meth2  TEST-DATA_fm_curves.png  TEST-DATA_pr_curves.png
        
# (cv549) huyentn2@cig-01:~/huyen/project/project_nano_count$ cd han/


# C:\Users\Admin>scp -r cig:/home/huyentn2/huyen/project/data_nano/han/ D:\cig1_backup\
        
# C:\Users\Admin>scp -r cig:/home/huyentn2/huyen/project/project_nano_count/han/ D:\cig1_backup\
        
# C:\Users\Admin>scp -r cig:/home/huyentn2/huyen/cell_counting/objects_counting_dmap_JUNO/output/particle_400_crop_360 D:\cig1_backup\objects_counting_dmap_JUNO\output\
        

# python get_weinan_ct_json.py --fold 1 --d particle