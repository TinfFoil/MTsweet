import pandas as pd
import argparse, torch
from hlepor import single_hlepor_score
from bert_score import BERTScorer
from comet import download_model, load_from_checkpoint
from transquest.algo.sentence_level.monotransquest.run_model import MonoTransQuestModel

def main(src, ref, trg, tsv_path):

    filename = tsv_path.split('/')[-1]
    path = "/".join(tsv_path.split('/')[:-1])

    current_tsv = pd.read_csv(tsv_path, sep='\t')
    translation_dict = {idx: (row[src], row[ref], row[trg]) for idx, row in current_tsv.iterrows()}
    current_tsv["scores"] = [[] for n in range(len(current_tsv["scores"]))]
    
    
    print("\nLoading COMET...\n")
    comet_path = download_model("wmt20-comet-da")
    comet = load_from_checkpoint(comet_path)
    print("\nLoading TransQuest...\n")
    transquest = MonoTransQuestModel("xlmroberta", "TransQuest/monotransquest-da-en_de-wiki", num_labels=1, use_cuda=torch.cuda.is_available())
    print("\nLoading BERTScore...\n")
    bert = BERTScorer(lang="de", rescale_with_baseline="True")
    
    print("\nComputing scores...\n")
    for idx, translations in translation_dict.items():
        hLepor_value = single_hlepor_score(translations[1], translations[2], alpha=2.95, beta=2.68, n=2, weight_elp=1.0, weight_pos=11.79, weight_pr=1.87)
        hLepor_value = round(hLepor_value, 4)
        
        P, R, BERTScore_value = bert.score([translations[2]], [translations[1]]) 
        prediction, sys_prediction = comet.predict([{"src": translations[0], "mt": translations[2], "ref": translations[1]}], batch_size=8, gpus=1)
        transquest_predictions, raw_outputs = transquest.predict([[translations[2], translations[0]]])

        current_tsv.at[idx, 'scores'].extend([hLepor_value, round(BERTScore_value.item(), 4), round(prediction[0], 4), round(transquest_predictions.tolist(), 4)])
    
    current_tsv.to_csv(f"{path}/scored_{filename}", sep="\t", index=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', required=True, type=str, metavar='FILEPATH', help='The PATH to the TSV file')
    parser.add_argument('--src', required=True, type=str, metavar='SOURCE', help='The source column')
    parser.add_argument('--ref', required=True, type=str, metavar='REFERENCE', help='The reference translation column')
    parser.add_argument('--trg', required=True, type=str, metavar='TARGET', help='The target translation column')

    args = parser.parse_args()
    main(args.src, args.ref, args.trg, args.filepath)
