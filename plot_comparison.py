import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pdb
import os


LOG_DIR = "/home/huyentn2/huyen/project/Weinan_prediction/out_wein/particle/"

def plot_compare(fold, MAE1, MAE2, path1, path2, code_name, true_p):
    # f = open('/home/huyentn2/huyen/cell_counting/objects_counting_dmap/output_log/true_count_fold{}.json'.format(fold))

    # f = open(true_p)
    f = open("/home/huyentn2/huyen/copy_cig/cell_counting/log_result/model1/particle/{}/true_count_fold{}.json".format(fold, fold))
    data = json.load(f)

    # f = open(path1)
    f = open("/home/huyentn2/huyen/project/Weinan_prediction/out_wein/particle/{}/wein_count_fold{}.json".format(fold, fold))
    data_pred = json.load(f)


    # f = open(path2)
    f = open("/home/huyentn2/huyen/copy_cig/cell_counting/log_result/model1/particle/{}/prediction_fold{}.json".format(fold, fold))
    data_pred2 = json.load(f)


    idx2name = {}
    count_GT = []
    count_ = []
    count_2 = []

    # #Method 1: plot in the order in the dataset
    # for i, file in enumerate(data):
    #     idx2name[i] = file
    #     count_GT.append(data[file])
    #     count_.append(data_pred[file])
    #     count_2.append(data_pred2[file])

    #Method 2: plot in the order of biggest number to smallest
    sorted_data = {os.path.basename(k): v for k, v in sorted(data.items(), key=lambda item: item[1])}

    data_pred2 = {os.path.basename(k): v for k, v in sorted(data_pred2.items(), key=lambda item: item[1])}

    # pdb.set_trace()
    
    for i, file in enumerate(sorted_data):

        # pdb.set_trace()
        idx2name[i] = file
        count_GT.append(sorted_data[file])
        count_.append(data_pred[file])
        count_2.append(data_pred2[file])


    count_GT = np.array(count_GT)
    count_ = np.array(count_)
    count_2 = np.array(count_2)

    print("Average percentage MSE of UNet method ", np.sum(abs(count_2 - count_GT) / count_GT)/ len(count_GT))
    print("Average percentage MSE of handcraft method ", np.sum(abs(count_ - count_GT) / count_GT)/ len(count_GT))
    print("Absolute MSE of handcraft method ", np.sum(abs(count_ - count_GT))/ len(count_GT))


    MAE1 = np.sum(abs(count_ - count_GT) / count_GT)/ len(count_GT)
    MAE2 = np.sum(abs(count_2 - count_GT) / count_GT)/ len(count_GT)
    mse1 = np.sum(abs(count_ - count_GT))/ len(count_GT)

    # pdb.set_trace()


    fig = plt.figure()
    ax = plt.axes()
    x = np.linspace(0, len(count_GT)-1, len(count_GT))
    ax.plot(x, count_GT, 'r', label="ground_truth")

    ax.plot(x, count_, label= "method " + str(code_name) + " MAE: " + "{0:.2f}% ".format(MAE1) + "{0:.2f}".format(mse1), linestyle='dashed')
    ax.plot(x, count_2, label= "method " + "UNet" + " MAE: " + "{0:.2f}%".format(MAE2), linestyle='dashed')
    ax.legend()

    plt.xlabel("Image index")
    plt.ylabel("Num of nano-particles")
    # fig.savefig(save_dir + input_dir.split("/")[-2] + '.png')

    fig.savefig(os.path.join(LOG_DIR, '{}/comparison_fold_{}.png'.format(fold, fold)))


    # print("Index to image name dictionary: ", idx2name)

    # with open("/home/huyentn2/huyen/project/Weinan_prediction/idx2name_evaluated_img_fold{}.json".format(fold), "w") as outfile:
    #     json.dump(idx2name, outfile)    


    # pdb.set_trace()



def print_error():

    ave_MSE = []
    ave_MAE = []
    ave_MRE = []

    for fold in range(1,11):

        f = open("/home/huyentn2/huyen/project/Weinan_prediction/out_wein/particle/{}/true_count_fold{}.json".format(fold, fold))
        data_true = json.load(f)    
        
        f = open("/home/huyentn2/huyen/project/Weinan_prediction/out_wein/particle/{}/wein_count_fold{}.json".format(fold, fold))
        data_wein = json.load(f)

        true_list = []
        wein_list = []
        for k in list(data_true.keys()):
            img_name = os.path.basename(k)
            true_list.append(data_true[k])
            wein_list.append(data_wein[img_name])

        err = np.array(true_list) - np.array(wein_list)

        res = {}
        res['MRE'] = np.mean(np.abs(err)/ np.array(true_list))
        res['MAE'] = np.mean(np.abs(err))
        res['MSE'] = np.sqrt(np.mean(np.array(err)**2))

        print("{:.2f}     {:.2f}     {:.2f}".format(res['MSE'], res['MAE'], res['MRE']))

        ave_MSE.append(np.sqrt(np.mean(np.array(err)**2)))
        ave_MAE.append(np.mean(np.abs(err)))
        ave_MRE.append(np.mean(np.abs(err)/ np.array(true_list)))

        with open("/home/huyentn2/huyen/project/Weinan_prediction/out_wein/particle/{}/res_metric{}.json".format(fold, fold), "w") as outfile:
            json.dump(res, outfile)


    print("Average all folds:")
    print("MSE: ", np.mean(np.array(ave_MSE)))
    print("MAE: ", np.mean(np.array(ave_MAE)))
    print("MRE: ", np.mean(np.array(ave_MRE)))






if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type = int, dest='fold', default= 1)
    parser.add_argument('--M1', type = float, dest='MAE1', default= 0)
    parser.add_argument('--M2', type = float, dest='MAE2', default= 0)
    parser.add_argument('--p1', type = str, dest='path1', default='')   # tradtional
    parser.add_argument('--p2', type = str, dest='path2', default='')  # UNet
    parser.add_argument('--code_name', type = str, dest='code_name', default='hand-craft')
    parser.add_argument('--true_p', type = str, dest='true_p', default='')
    args = parser.parse_args()


    # true_p = '/home/huyentn2/huyen/project/project_nano_count/han/predictions_UNet/output_log/true_count_fold{}.json'.format(args.fold)


    # plot_compare(args.fold, args.MAE1, args.MAE2, args.path1, args.path2, args.code_name, args.true_p)

    print_error()





# (cv549) huyentn2@cig-01:~$ python plot_comparison.py --fold 2 --p1 /home/huyentn2/huyen/project/project_nano_count/han/han_fold2_T130.json --p2 /home/huyentn2/huyen/ce
# ll_counting/objects_counting_dmap/output_log/prediction_fold2.json


# (cv549) huyentn2@cig-01:~/huyen/project/project_nano_count/han$ python plot_comparison.py --fold 2 --p1 /home/huyentn2/huyen/project/project_nano_count/han/han_method_output/best_outputs/han_fold2_T130.json --p2 /home/huyentn2/huyen/cell_counting/objects_counting_dmap/output_log/prediction_fold2.json


#  python plot_comparison.py --fold 1 --p1 /home/huyentn2/huyen/project/project_nano_count/han/weinan_fold1.json --p2 /home/huyentn2/huyen/cell_counting/objects_counting_dmap/output_log/prediction_fold1.json


# python plot_comparison.py --fold 1 --p1 /home/huyentn2/huyen/project/project_nano_count/han/weinan_fold1.json --p2 /home/huyentn2/huyen/project/project_nano_count/han/predictions_UNet/output_log/prediction_fold1.json


# python plot_comparison.py --fold 2 --p1 /home/huyentn2/huyen/project/project_nano_count/han/weinan_fold2.json --p2 /home/huyentn2/huyen/project/project_nano_count/han/predictions_UNet/output_log/prediction_fold2.json






# python plot_comparison.py --fold 10 --p1 /home/huyentn2/huyen/project/project_nano_count/han/weinan_fold10.json --p2 /home/huyentn2/huyen/project/project_nano_count/han/predictions_UNet/output_log/prediction_fold10.json



# (nano) huyentn2@cig-03:~$ python huyen/project/Weinan_prediction/plot_comparison.py --p1 /home/huyentn2/huyen/project/Weinan_prediction/wein_count_fold1.json --p2 /home/huyentn2/huyen/project/Weinan_prediction/prediction_fold1.json  --true_p /home/huyentn2/huyen/project/Weinan_prediction/true_count_fold1.json

# python plot_comparison.py --p1  --p2  --true_p