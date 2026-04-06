import argparse
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data.txt")
    parser.add_argument("--output", type=str, default="data_clean.txt")
    return parser.parse_args()


def clean_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    cleaned_lines = []
    for line in text.split("\n"):
        cleaned_lines.append(line.strip())

    cleaned = "\n".join(cleaned_lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned + "\n"


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} does not exist")

    raw_text = input_path.read_text(encoding="utf-8")
    cleaned_text = clean_text(raw_text)
    output_path.write_text(cleaned_text, encoding="utf-8")

    print("cleaning complete")
    print("input file :", input_path)
    print("output file:", output_path)
    print("raw chars  :", len(raw_text))
    print("clean chars:", len(cleaned_text))
    print("raw lines  :", len(raw_text.splitlines()))
    print("clean lines:", len(cleaned_text.splitlines()))


if __name__ == "__main__":
    main()
