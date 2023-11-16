from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score

import argparse
import os
from pathlib import Path
from tqdm import tqdm
import json
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

def draw_roc(labels, preds, file):
    '''
    labels: list
    preds: list
    '''
    colors = ['r','b','g','y','pink']
    fpr, tpr, thersholds = roc_curve(labels, preds, pos_label=1) # pos_label指定哪个标签为正样本
    roc_auc = auc(fpr, tpr)  # 计算ROC曲线下面积


    plt.plot(fpr, tpr, '-', color=colors[4], label='ROC (area=%.6f)' % (roc_auc), lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    print(file.split('_')[-1], colors[4], roc_auc)

def draw_pr(labels, preds, file):
    '''
    labels: list
    preds: list
    '''
    colors = ['r','b','g','y','pink']
    precision, recall, thersholds = precision_recall_curve(labels, preds, pos_label=1) # pos_label指定哪个标签为正样本
    area = average_precision_score(labels, preds, pos_label=1)  # 计算PR曲线下面积

    with open('inference_output/PR_thersholds_3000.txt','w') as rf:
        for i in range(len(thersholds)):
            rf.write('\t'.join([str(precision[i+1]), str(recall[i+1]), str(thersholds[i])]) + '\n')
    plt.plot(recall, precision, '-', color=colors[4], label='PR (area=%.6f)' % (area), lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    print(file.split('_')[-1], colors[4], area)

gambling_tool = [
                'wheel',
                'dice',
                'spin',
                'stone',
                'pool',
                'sand',
                'ball',
                'bead',
                'number',
                'box'
                 ]
def gambling_post_process(result_file,datapath,country):
    # if os.path.isfile(datapath):
    #     dataset = open(datapath,'r').readlines()
    # else:
    #     file_list = list((Path(datapath)).rglob('*'))
    #     dataset = []
    #     for filepath in tqdm(file_list):
    #         print(filepath)
    #         dataset += open(filepath,'r').readlines()
    with open(result_file) as f:
        result = f.readlines()
        gt_label = []
        predict_abel = []
        for i,line in enumerate(result):
            if 'ERROR DATA' in line:
                continue
            # coun = json.loads(dataset[i])['country']
            # if coun == country or country == 'all':
            # else:
            result_string = line.lower().split('    ')[1]
            gt_label_this = int(line.lower().split('    ')[0])
            gt_label.append(gt_label_this)
            flag = 1
            if 'yes' in result_string:
                predict_abel.append(1)
                flag=0
            # elif 'contains gambling elements' in result_string:
            #     predict_abel.append(1)
            #     flag=0
            # elif 'contains elements of gambling' in result_string:
            #     predict_abel.append(1)
            #     flag=0
            # elif 'gambling scenario' in result_string:
            #     predict_abel.append(1)
            #     flag=0
            else:
                for item in gambling_tool:
                    if item in result_string:
                        predict_abel.append(1)
                        flag=0
                        break
            if flag:
                predict_abel.append(0)
    return gt_label, predict_abel

def nppc_post_process(result_file,datapath,country):
    # if os.path.isfile(datapath):
    #     dataset = open(datapath,'r').readlines()
    # else:
    #     file_list = list((Path(datapath)).rglob('*'))
    #     dataset = []
    #     for filepath in tqdm(file_list):
    #         print(filepath)
    #         dataset += open(filepath,'r').readlines()
    with open(result_file) as f:
        result = f.readlines()
        gt_label = []
        predict_abel = []
        for i,line in enumerate(result):
            if 'ERROR DATA' in line:
                continue
            # coun = json.loads(dataset[i])['country']
            # if coun == country or country == 'all':
            # else:
            result_string = line.lower().split('\t')[3].strip()
            gt_label_this = int(line.lower().split('\t')[2])
            gt_label.append(gt_label_this)
            flag = 1
            if  'yes' in result_string:
                predict_abel.append(1)
                flag=0
            if flag:
                predict_abel.append(0)
    return gt_label, predict_abel
def mfe_post_process(result_file,datapath,country):
    # if os.path.isfile(datapath):
    #     dataset = open(datapath,'r').readlines()
    # else:
    #     file_list = list((Path(datapath)).rglob('*'))
    #     dataset = []
    #     for filepath in tqdm(file_list):
    #         print(filepath)
    #         dataset += open(filepath,'r').readlines()
    with open(result_file) as f:
        result = f.readlines()
        gt_label = []
        predict_abel = []
        predict_score = []
        for i,line in enumerate(result):
            if 'ERROR DATA' in line:
                continue
            # coun = json.loads(dataset[i])['country']
            # if coun == country or country == 'all':
            # else:
            result_string = line.lower().split('\t')[2].strip()
            gt_label_this = line.lower().split('\t')[1]
            if '1' in gt_label_this or '0' in gt_label_this:
                gt_label_this = int(gt_label_this)
            else:
                gt_label_this = 1 if 'yes' in gt_label_this else 0
            gt_label.append(gt_label_this)
            flag = 1
            if  float(result_string)>0.4:
            # if  'yes' in result_string:
                predict_abel.append(1)
            else:
                predict_abel.append(0)
            predict_score.append(float(result_string))
    return gt_label, predict_abel, predict_score
def eval(args):
    if args.task =='gambling':
        gt_label, predict_abel = gambling_post_process(args.result_file, args.data_path, args.country)
    elif args.task =='nppc':
        gt_label, predict_abel = nppc_post_process(args.result_file, args.data_path, args.country)
    elif args.task =='mfe':
        result_file_list = ["inference_output/gandalf_feature_3000.txt"]
        result_data = [mfe_post_process(result_file, args.data_path, args.country) for result_file in result_file_list]
        plt.figure(figsize=(10,7), dpi=300)
        [draw_roc(gt_label, predict_score, result_file_list[i]) for i,(gt_label,_, predict_score) in enumerate(result_data)]
        plt.legend(result_file_list)
        plt.savefig('inference_output/valley_mfe_gandalf_vector_ROC.png')

        plt.figure(figsize=(10,7), dpi=300)
        [draw_pr(gt_label, predict_score, result_file_list[i]) for i,(gt_label,_, predict_score) in enumerate(result_data)]
        plt.legend(result_file_list)
        plt.savefig('inference_output/valley_mfe_gandalf_vector_PR.png')
        gt_label, predict_abel,_ = result_data[0]

    print('acc:'+str(accuracy_score(gt_label, predict_abel)))
    print('recall:'+str(recall_score(gt_label, predict_abel)))
    print('precision:'+str(precision_score(gt_label, predict_abel)))
    print('f1_score:'+str(f1_score(gt_label, predict_abel)))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file", type=str, default="inference_output/gandalf_feature_3000.txt")
    parser.add_argument("--task", type=str, default="mfe")
    parser.add_argument("--data-path", type=str, default="/mnt/bn/luoruipu-disk/meta_data/sft_data/gambling/GB/test")
    parser.add_argument("--country", type=str, default="MY",choices=['GB','ID','MY','TH','all'])
    args = parser.parse_args()
    eval(args)