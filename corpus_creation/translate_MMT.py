from modernmt import ModernMT
import pandas as pd
import argparse
import time

def translate_it(system, src, trg, string_list):
    translated = system.translate(src, trg, string_list)
    translated = [translation.__dict__ for translation in translated]
    return translated


def get_key(key_path):
    with open(key_path, 'r') as f:
        key = f.read()
        key = key.strip('\n')
    return key

def main(key_path, src, trg, tsv_path):
    API_KEY = get_key(key_path)
    mmt = ModernMT(API_KEY)
    filename = tsv_path.split('/')[-1]

    current_tsv = pd.read_csv(tsv_path, sep='\t', index_col = 0).sample(frac=1).reset_index(drop=True)


    size = 70
    list_of_dfs = [current_tsv.loc[i:i+size-1,:] for i in range(0, len(current_tsv),size)]
    char_count = 0
    for idx, df in enumerate(list_of_dfs):
        char_count += df[src].apply(lambda x: len(x)).sum()
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
        source_list = [row[src.casefold()] for idx, row in df.iterrows()]
        translation = translate_it(mmt, src.casefold(), trg.casefold(), source_list)
        target_list = [MT['translation'] for MT in translation]
        df['MT'] = target_list
        
        translated.append(df)
        total_dfs -= 1
       
         
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
