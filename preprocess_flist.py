import os
import argparse
from tqdm import tqdm
from random import shuffle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", type=str, default="./filelists/train.txt", help="path to train list")
    parser.add_argument("--val_list", type=str, default="./filelists/val.txt", help="path to val list")
    parser.add_argument("--test_list", type=str, default="./filelists/test.txt", help="path to test list")
    parser.add_argument("--source_dir", type=str, default="./dataset/vctk-16k", help="path to source dir")
    args = parser.parse_args()
    
    train = []
    val = []
    test = []
    idx = 0
    
    for speaker in tqdm(os.listdir(args.source_dir)):
        wavs = os.listdir(os.path.join(args.source_dir, speaker))
        shuffle(wavs)
        train += wavs[2:-10]
        val += wavs[:2]
        test += wavs[-10:]
        
    shuffle(train)
    shuffle(val)
    shuffle(test)
            
    print("Writing", args.train_list)
    with open(args.train_list, "w") as f:
        for fname in tqdm(train):
            speaker = fname[:4]
            wavpath = os.path.join("DUMMY", speaker, fname)
            f.write(wavpath + "\n")
        
    print("Writing", args.val_list)
    with open(args.val_list, "w") as f:
        for fname in tqdm(val):
            speaker = fname[:4]
            wavpath = os.path.join("DUMMY", speaker, fname)
            f.write(wavpath + "\n")
            
    print("Writing", args.test_list)
    with open(args.test_list, "w") as f:
        for fname in tqdm(test):
            speaker = fname[:4]
            wavpath = os.path.join("DUMMY", speaker, fname)
            f.write(wavpath + "\n")
            