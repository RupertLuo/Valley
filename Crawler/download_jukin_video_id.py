import requests
import json as js
import math
from argparse import ArgumentParser

def main(args):
    headers = {
        "X-Algolia-Api-Key": "a6099f9d3771d6ceb142321ac5273d16",
        "X-Algolia-Application-Id": "XSWHBQ6C6E",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    category_number = {
    "Fails": 10000,  "Pets": 10000,  "Awesome": 9507,  "Wildlife": 8896,  "Humor": 6493,  "Talent": 5471,
    "DIY": 2569,  "Uplifting": 2431,  "Newsworthy": 1957,  "Cute": 1952,  "Parenting": 1880,  "Weather": 1630,
    "Fitness": 1385,  "Family": 1296,  "Art": 1154,  "Food": 1116,  "Crashes": 980,  "Sports": 947,  "Vehicles": 439,
    "Lifestyle": 370,  "Nature": 330,  "Travel": 294,  "Crime": 161,  "Paranormal": 115,  "RecordSetter": 3,  "Nitro Circus": 1
    }

    sum_data = 0
    for key in category_number:
        sum_data+=category_number[key]
    print('number of all vid: ',sum_data)

    result_number = dict()
    for category in category_number:
        page_number = math.ceil(category_number[category]/1000)
        data = []
        for i in range(page_number):
            json_data = {"query":"","userToken":"guest","hitsPerPage":1000,"page":i,"facets":["category"],"facetFilters":[["category:"+category]]}
            a = requests.post("https://xswhbq6c6e-2.algolianet.com/1/indexes/public_lp/query", headers=headers, json=json_data
            )
            data+=js.loads(a.content)['hits']
            result_number[category] = data


    js.dump(result_number,open(args.savefolder+'/'+'jukin-100k.json','w'))

if __name__ == "__main__":
    parser = ArgumentParser(description="Script to parallel download jukinmedia video")
    parser.add_argument("--savefolder", default='./jukinmedia',)
    args = parser.parse_args()
    main(args)