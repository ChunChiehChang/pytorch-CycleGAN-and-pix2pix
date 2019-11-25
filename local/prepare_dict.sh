#!/usr/bin/env bash

data_dir=data

. ./local/parse_options.sh || exit 1;

dir=$data_dir/local/dict
mkdir -p $dir

cat $data_dir/trainA/text $data_dir/trainB/text $data_dir/testA/text $data_dir/testB/text | cut -d' ' -f2- | tr ' ' '\n' | \
    sort -u | sed '/^$/d' | \
    python3 -c \
    'import sys, io, unicodedata; \
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf8"); \
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf8"); \
    [sys.stdout.write(unicodedata.normalize("NFKC",line).strip() + " " + " ".join(list(unicodedata.normalize("NFKC", line).strip())) + "\n") for line in sys.stdin];' > $dir/lexicon.txt

cut -d' ' -f2- $dir/lexicon.txt | tr ' ' '\n' | sort -u | sed '/^$/d' >$dir/phones.txt
