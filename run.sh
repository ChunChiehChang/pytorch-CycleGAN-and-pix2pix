#!/bin/bash

. ./path.sh
. ./cmd.sh

data=datasets/casia_synth
dir=casia_dir
stage=-1

. ./local/parse_options.sh || exit 1;

#if [ ${stage} -le 1 ]; then
#    # Process text
#    local/prepare_dict.sh --data-dir $data
#fi

if [ ${stage} -le 2 ]; then
    mkdir -p $dir
    phone_size=$(wc -l $data/local/dict/phones.txt | cut -d' ' -f1)
    $cmd --gpu 1 --mem 8G $dir/train.log limit_num_gpus.sh python3 train.py \
        --dataroot $data \
        --name casia_synth_ctc \
        --dataset_mode ctc \
        --model cycle_gan_ctc \
        --netG resnet_9blocks \
        --netD basic \
        --input_nc 1 \
        --output_nc 1 \
        --preprocess none \
        --no_flip \
        --niter 50 \
        --niter_decay 50 \
        --ctc_embedding_size 60 \
        --ctc_phone_size $phone_size
fi
