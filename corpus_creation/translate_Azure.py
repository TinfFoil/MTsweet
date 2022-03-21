import pandas as pd
import argparse, requests, uuid
import numpy as np
import time


def get_key(key_path):
    with open(key_path, 'r') as f:
        key = f.read()
        key = key.strip('\n')
    return key


def main(key_path, src, trg, tsv_path):
    API_KEY = get_key(key_path)
    endpoint = "https://api.cognitive.microsofttranslator.com"
    path = "/translate"
    params = {'api-version':'3.0', 'from': f'{src}', 'to': [trg]}
    location = "westeurope"
    constructed_url = endpoint + path
    headers = {
            'Ocp-Apim-Subscription-Key': API_KEY,
            'Ocp-Apim-Subscription-Region': location,
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
            }
    filename = tsv_path.split('/')[-1]

    current_tsv = pd.read_csv(tsv_path, sep='\t')
    
    size = 70
    list_of_dfs = [current_tsv.loc[i:i+size-1,:] for i in range(0, len(current_tsv),size)]
    char_count = 0
    for idx, df in enumerate(list_of_dfs):
        char_count += df['de'].apply(lambda x: len(x)).sum()
        if char_count < 666666:
            pass
        else:
            list_of_dfs = list_of_dfs[:idx]
            break


    translated = []
    total_dfs = len(list_of_dfs)

    for df in list_of_dfs:
        time.sleep(35)
        print(f'{total_dfs} missing', flush=True)
        body = [{'text' : row[src.casefold()]} for idx, row in df.iterrows()]
        request = requests.post(constructed_url, params=params, headers=headers, json=body)
        response = request.json()
        print(response)

        target_list = [MT['translations'][0]['text'] for MT in response]

        df['MT'] = target_list
        
        total_dfs -= 1
        translated.append(df)
        
    
    translated_file = pd.concat(translated)
    translated_file.to_csv(f"translated_{filename}", sep="\t", index=False)
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', required=True, type=str, metavar='FILEPATH', help='The PATH to the TSV file')
    parser.add_argument('--src', required=True, type=str, metavar='SOURCE', help='The source language of the translation')
    parser.add_argument('--trg', required=True, type=str, metavar='TARGET', help='The target language of the translation')
    parser.add_argument('--key', required=True, type=str, metavar='APIKEY', help='The PATH to a .txt file containing the API Key for ModernMT')

    args = parser.parse_args()
    main(args.key, args.src, args.trg, args.filepath)
