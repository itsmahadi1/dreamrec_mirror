import os
import argparse
import pandas as pd
from collections import defaultdict

# ---- Minimal data_partition (copied from CARCA) ----
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train, user_valid, user_test = {}, {}, {}
    # assume user/item index starting from 1
    with open('./Data/%s.txt' % fname, 'r') as f:
        for line in f:
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = [User[user][-2]]
            user_test[user] = [User[user][-1]]

    return [user_train, user_valid, user_test, usernum, itemnum]
# ----------------------------------------------------


def convert_carca_to_dreamrec(raw_dataset, out_dir, max_cap=200):
    """
    Convert CARCA dataset into DreamRec format (sequential only).

    Args:
        raw_dataset (str): dataset name (expects ./Data/{name}.txt)
        out_dir (str): output directory for DreamRec (e.g. ./data/beauty)
        max_cap (int): cap for maximum sequence length
    """

    user_train, user_valid, user_test, usernum, itemnum = data_partition(raw_dataset)
    print(f"[INFO] Users: {usernum}, Items: {itemnum}")

    # Find the longest sequence length and cap it
    longest_seq = max(len(seq) for seq in list(user_train.values()) +
                                      list(user_valid.values()) +
                                      list(user_test.values()))
    seq_size = min(longest_seq, max_cap)
    print(f"[INFO] Using seq_size = {seq_size}")

    def build_samples(user_dict):
        seqs, lens, targets = [], [], []
        for _, items in user_dict.items():
            if len(items) < 2:
                continue
            for idx in range(1, len(items)):
                seq = items[:idx][-seq_size:]  # last seq_size
                target = items[idx]
                padded_seq = [itemnum] * (seq_size - len(seq)) + seq
                seqs.append(padded_seq)
                lens.append(len(seq))
                targets.append(target)
        return pd.DataFrame({"seq": seqs, "len_seq": lens, "next": targets})

    train_df = build_samples(user_train)
    valid_df = build_samples(user_valid)
    test_df  = build_samples(user_test)

    os.makedirs(out_dir, exist_ok=True)
    train_df.to_pickle(os.path.join(out_dir, "train_data.df"))
    valid_df.to_pickle(os.path.join(out_dir, "valid_data.df"))
    test_df.to_pickle(os.path.join(out_dir, "test_data.df"))

    pd.DataFrame({
        "seq_size": [seq_size],
        "item_num": [itemnum]
    }).to_pickle(os.path.join(out_dir, "data_statis.df"))

    print(f"[INFO] Saved DreamRec dataset to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (expects ./Data/{dataset}.txt)")
    parser.add_argument("--out", type=str, default="./data",
                        help="Output base directory for DreamRec")
    parser.add_argument("--max_cap", type=int, default=200,
                        help="Cap for maximum sequence length")
    args = parser.parse_args()

    raw_dataset = args.dataset
    out_dir = os.path.join(args.out, raw_dataset.lower())
    convert_carca_to_dreamrec(raw_dataset, out_dir, max_cap=args.max_cap)
