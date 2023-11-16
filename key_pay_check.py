import requests

def get_usage(key):
    queryUrl = 'https://api.openai.com/dashboard/billing/subscription'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Authorization': f'Bearer {key}',
        'Accept': '*/*',
        'Host': 'api.openai.com',
        'Connection': 'keep-alive'
    }
    
    r = requests.get(queryUrl, headers=headers)
    return r.json()


print("key:", get_usage("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"))