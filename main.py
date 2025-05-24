import os
import subprocess

def run_preprocess():
    print(">>> [1/4] 预处理 item_meta ...")
    os.system("python preprocess/process_item_meta.py")

def run_train():
    print(">>> [2/4] 训练模型 ...")
    os.system("python train.py")

def run_evaluate():
    print(">>> [3/4] 评估 Recall@10 ...")
    os.system("python evaluate.py")

def run_infer():
    print(">>> [4/4] 生成推荐列表 ...")
    os.system("python infer.py")

def main():
    run_preprocess()
    run_train()
    run_evaluate()
    run_infer()
    print(">>> 全流程完成！推荐已写入 output/submission.csv")

if __name__ == '__main__':
    main()