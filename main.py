import os
import subprocess

def run_preprocess():
    print(">>> [1/4] Preprocessing item_meta ...")
    os.system("python3 preprocess/process_item_meta.py")

def run_train():
    print(">>> [2/4] Training the model ...")
    os.system("python3 train.py")

def run_evaluate():
    print(">>> [3/4] Evaluating Recall@10 ...")
    os.system("python3 evaluate.py")
    os.system("python3 recall.py")

def run_infer():
    print(">>> [4/4] Generating recommendation list ...")
    os.system("python3 infer.py")

def main():
    run_preprocess()
    run_train()
    run_evaluate()
    run_infer()
    print(">>> Pipeline complete! Recommendations saved to output/submission.csv")

if __name__ == '__main__':
    main()