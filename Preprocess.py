import pandas as pd
import argparse

def split_train_test(csv_path, test_size=0.15, val_size=0.15, train_path="train.csv", val_path="val.csv", test_path="test.csv"):
    print("-- Reading csv file --")
    df = pd.read_csv(csv_path, index_col=0)
    # Split into train and val and test
    train = df.sample(frac=1-test_size-val_size, random_state=42)
    val = df.drop(train.index).sample(frac=val_size/(test_size+val_size), random_state=42)
    test = df.drop(train.index).drop(val.index)
    # Save to csv
    print("-- Saving train csv --")
    train.to_csv(train_path)
    print("-- Saving validation csv --")
    val.to_csv(val_path)
    print("-- Saving test csv --")
    test.to_csv(test_path)
    return train, val, test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into train, val, and test sets')
    parser.add_argument('-f', '--csv_path', default="./data/binding_affinity_data.csv", help='path to the input csv file')
    parser.add_argument('-t', '--test_size', type=float, default=0.15, help='test set size')
    parser.add_argument('-v', '--val_size', type=float, default=0.15, help='validation set size')
    parser.add_argument('-tp', '--train_path', default="./data/train.csv", help='path to save train set')
    parser.add_argument('-vp', '--val_path', default="./data/val.csv", help='path to save validation set')
    parser.add_argument('-tep', '--test_path', default="./data/test.csv", help='path to save test set')
    args = parser.parse_args()

    split_train_test(args.csv_path, args.test_size, args.val_size, args.train_path, args.val_path, args.test_path)