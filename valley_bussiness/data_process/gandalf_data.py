import pandas as pd
import json as js 
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path

def process_feature(general_feature_df, gandalf_feature, train_vid):
    number_row = general_feature_df.shape[0]
    assert number_row == gandalf_feature.shape[0]
    result = []
    for i in tqdm(range(0, number_row)):
        general_feature_dict = dict(general_feature_df.iloc[i])
        general_feature_dict['vid'] = str(general_feature_dict['vid'])
        if general_feature_dict['vid'] not in train_vid:
            continue
        else:
            general_feature_dict['video_desp'] = None if str(general_feature_dict['video_desp']) == 'nan' else str(general_feature_dict['video_desp'])
            general_feature_dict['asr'] = None if str(general_feature_dict['asr']) == 'nan' else str(general_feature_dict['asr'])
            general_feature_dict['merge_ocr'] = None if str(general_feature_dict['merge_ocr']) == 'nan' else str(general_feature_dict['merge_ocr'])
            general_feature_dict['reject_reason'] = None if str(general_feature_dict['reject_reason']) == 'nan' else str(general_feature_dict['reject_reason'])
            general_feature_dict['audit_result_code'] = int(general_feature_dict['audit_result_code'])
            
            gandalf_feature_vector = gandalf_feature[i].tolist()
            general_feature_dict['gandalf_vector'] = gandalf_feature_vector
            result.append(general_feature_dict)
    return result

def process_data(data, train_vid, save_path, index, gandalf_feature_list):
    rf = open(save_path+f'.part{index}','w')
    data = data.replace('\\N',-1.0)
    add_feature = [i for i in data.columns if i not in drop_feature]
    # 填充na
    new_data_orig = pd.DataFrame(data, columns=add_feature)
    values = {key: -1.0 for key in gandalf_feature_list}
    new_data = new_data_orig.iloc[:,9:].fillna(value=values,inplace=False)
    
    feature_matrix = new_data.values.astype(np.float32)
    # courtry data
    country_dummy = pd.get_dummies(data['country']).astype(np.float32)
    country_matrix = country_dummy.values
    # feature concat
    feature_final = np.concatenate((feature_matrix,country_matrix),axis=1)
    general_feature = new_data_orig[['vid','anchor_title','video_desp','audit_result_code','country','asr','merge_ocr']] 
    reject_reason = data[['reject_reason']]
    general_feature_df = pd.concat([general_feature,reject_reason],axis=1)
    result = process_feature(general_feature_df, feature_final, train_vid)
    js.dump(result, rf, ensure_ascii=False)

def gather_data(save_path):
    save_path = Path(save_path)
    path_list = list(save_path.parent.glob(save_path.name+'.*'))
    result = []
    for path in tqdm(path_list):
        try:
            result+=js.load(open(path))
            path.unlink()
        except Exception as e:
            print(e)
    js.dump(result, open(save_path,'w'), ensure_ascii=False)

if __name__ == '__main__':
    # 载入一个csv文件
    train_vid = js.load(open('/mnt/bn/ecom-lxy/zhuhe/valley/gpt_data/small_mfe_test_data_vid.json'))
    csv_path_list = Path('/mnt/bn/ecom-lxy/zhuhe/valley/gpt_data/new_ori_data/').rglob('*')
    already_file = [data.name[:-5] for data in list(Path('/mnt/bn/luoruipu-disk/meta_data/sft_data/mfe_gandalf_test').rglob('*.json'))]
    print(already_file)
    for csv_path in csv_path_list:
        if csv_path.name[12:14] == '10' and csv_path.name[:-4] not in already_file:
            print(f'processes file {csv_path.name}')
            save_path = f"/mnt/bn/luoruipu-disk/meta_data/sft_data/mfe_gandalf_test/{csv_path.name[:-4]+'.json'}"
            drop_feature = js.load(open('/mnt/bn/ecom-lxy/zhuhe/valley/gpt_data/drop_features.json'))
            general_feature_list = js.load(open('/mnt/bn/luoruipu-disk/code_base/valley/valley_bussiness/data_process/general_feature.json'))
            gandalf_feature_list = js.load(open('/mnt/bn/luoruipu-disk/code_base/valley/valley_bussiness/data_process/gandalf_feature.json'))
            df_iterator = pd.read_csv(csv_path, chunksize = 5000, low_memory=False)
            p = Pool(int(16))  # 指定进程池中的进程数
            for index,data in enumerate(tqdm(df_iterator)):
                p.apply_async(process_data, args = (data,train_vid, save_path, index, gandalf_feature_list))
                # process_data(data,train_vid, save_path, index)
                # break 
            p.close()
            p.join() 
            print(f'{csv_path.name} All subprocesses done.')
            gather_data(save_path)