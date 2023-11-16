import json as js
import re
test_data = js.load(open('/mnt/bn/ecom-lxy/zhuhe/valley/gpt_data/small_mfe_test_data.json'))
error_data = 0
for i,data in enumerate(test_data):
    try:
        country = re.findall(r'This video comes from (.*?) country', test_data[i]['conversations'][0]['value'])[0]
        description = re.findall(r'The description of this video is as follows: "(.*?)". \n\nThe title of this video is as follows', test_data[i]['conversations'][0]['value'])[0]
        asr = re.findall(r'The top 200 characters from this video voice is as follows: "(.*?)"', test_data[i]['conversations'][0]['value'])[0]
        test_data[i]['country'] = country
        desc_string = f'''The description of this video is as follows: "{description}". \n''' if description else '''The video does not have description. \n'''
        asr_string = f'''The person in this live video is saying: "{asr}". \n''' if asr else '''The person in this live video is say nothing. \n'''
        # title_string = f'''The title of the live video is: "{title}". \n'''
        # ocr_string = f'''The Optical character recognition information in this video is "{ocr}".\n'''
        q = '''Does this video have Misleading Functionality & Effect? \n<video>'''
        # all_tring = desc_string + asr_string + title_string + ocr_string +q
        all_tring = desc_string + asr_string +q
        test_data[i]['conversations'][0]['value'] = all_tring
        test_data[i]['video'] = test_data[i]['video'].replace('test_data','video_data')
    except Exception as e:
        error_data+=1
print(error_data)
js.dump(test_data, open('/mnt/bn/ecom-lxy/zhuhe/valley/gpt_data/small_mfe_test_data_old_prompt.json','w'))