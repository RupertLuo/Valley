from pathlib import Path
import json as js 
from tqdm import tqdm

def process_gandalf_row(line):
    '''{
        "id": "mfe_class_7286692623677492485", 
        "video": "/mnt/bn/ecom-lxy/zhuhe/valley/gpt_data/test_data/images/7286692623677492485", 
        "gt_label": 0, 
        "conversations": [
            {"from": "human", "value": "This video comes from PH country. 
            The description of this video is as follows: \"CB NIACINAMIDE SOAP FOR MEN 10x whitening soap whitening all skin types skin whitening for men #skinwhiteningformen #niacinamide #whiteningsoapformen #skinwhitening #fyp\u30b7\u309aviral \". 
            \n\nThe title of this video is as follows: \"CB NIACINAMIDE SOAP FOR MEN 10\". 
            \n\nThe top 200 characters extracted from this video frame is as follows: \"10X WHITENING SOAP FOR MEN Cosmi Beautil Ma 10xW Moisturizer Skin Firming Acne Journey v Niacinamide Fa Niacinamide Face bar Niacinamide soap irmming Moisturize Skin Firming BODY SOAP FOR MEN hitening\". 
            \n\nThe top 200 characters from this video voice is as follows: \"wag na wag kang gagamit ng niacinamide sa skin care mo kung ayaw mong mangyari ang mga sumusunod number one ayaw mong mag asawa char number one ayaw mo magmukhang fresh ang niacinamide ay nakakatulong\". 
            \n\nThis video may potentially violate one or multiple policies predicted by the classification model based on deep neural network. Here are the policy names: Body Impurities. 
            \n\nThe author of this video has registered for 5 days. The author in the past 30 days has published 8 e-commercial videos. The author in the past 30 days has published 9 videos. The e-commercial video ratio of this author is 0.89. The author to obtain the e-commercial permission by inviting by the operator. This author has gained 14.74 gmv in the past 30 days. This author has 5 fans. This author has 4 videos may potentially have risky, and 3 may violate the misleading functionality and effect policy. This author does publish 4 risky videos, and 0 violate the misleading functionality and effect policy. 
            \n\nDoes this video have Misleading Functionality & Effect? \n<video>"}, {"from": "gpt", "value": "No"}]
    '''
    title = line['anchor_title']
    desp = line['video_desp']
    asr = line['asr']
    ocr = line['merge_ocr']
    gandalf_vector = line['gandalf_vector']
    id = f"mfe_class_{line['vid']}"
    video = f"/mnt/bn/ecom-lxy/zhuhe/valley/gpt_data/video_data/images/{line['vid']}"
    gt_label = line['audit_result_code']==0 and line['reject_reason'] == 'Misleading Functionality and Effect'

    describe_query = f'''The description of this video is as follows: "{desp[:200]}". ''' if desp else '''The video doesn't have description. '''
    title_query = f'''\n\nThe title of this video is as follows: "{title}". ''' if title else '''\n\nThe video doesn't have title. '''
    ocr_query = f'''\n\nThe top 200 characters extracted from this video frame is as follows: "{ocr[:200]}". ''' if ocr else '''\n\nThe video can't extracted any characters. '''
    asr_query = f'''\n\nThe top 200 characters from this video voice is as follows: "{asr[:200]}". ''' if asr else '''\n\nThe video doesn't have any voice. '''
    gandalf_query = f'''\n\nThe another feature is as follows: <gandalf>'''
    conversations = [
        {'from': " human", "value": describe_query + title_query + ocr_query + asr_query + gandalf_query + "\n\nDoes this video have Misleading Functionality & Effect? \n<video>" },
        # {'from': " gpt", "value": "Yes" if gt_label else "No"}
    ]
    line_dict = dict(
        id = id,
        video = video,
        gt_label = gt_label,
        conversations = conversations,
        gandalf_vector = gandalf_vector
    )

    return line_dict


if __name__ == '__main__':
    gandalf_feature_dir = Path('/mnt/bn/luoruipu-disk/meta_data/sft_data/mfe_gandalf_test_str/')

    feature_file = list(gandalf_feature_dir.rglob('*.json'))

    result = []
    offset = 100000
    global_count = 0
    count = 0
    part = 0
    for f in tqdm(feature_file):
        lines = js.load(open(f))
        for line in lines:
            result.append(process_gandalf_row(line))
            count+=1
            global_count+=1
            if count > offset:
                js.dump(result, open(f'/mnt/bn/luoruipu-disk/meta_data/sft_data/mfe_gandalf_test/mfe_gandalf_train_part{part}.json','w'), ensure_ascii=False)
                part+=1
                count = 0
                result = []
    js.dump(result, open(f'/mnt/bn/luoruipu-disk/meta_data/sft_data/mfe_gandalf_test/mfe_gandalf_train_part{part}.json','w'), ensure_ascii=False)
    print(global_count)
    # data = js.load(open('/mnt/bn/luoruipu-disk/meta_data/sft_data/mfe_gandalf_train/mfe_gandalf_train_part5.json'))
    # print()
