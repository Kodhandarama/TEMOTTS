"""Pre-download every external asset (run on a node with internet access)."""
import argparse

from temotts.lm import LM_NAME
from temotts.vocoder import download_pwg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_dir", required=True)
    args = parser.parse_args()

    ckpt = download_pwg(args.asset_dir)
    print("PWG checkpoint:", ckpt)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    AutoTokenizer.from_pretrained(LM_NAME)
    AutoModelForSequenceClassification.from_pretrained(LM_NAME)
    print("LM cached:", LM_NAME)

    from g2p_en import G2p

    print("g2p_en check:", G2p()("temotts"))


if __name__ == "__main__":
    main()
