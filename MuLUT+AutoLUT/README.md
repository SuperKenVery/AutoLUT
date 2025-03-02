# AutoLUT applied on MuLUT
Original MuLUT: [repo](https://github.com/ddlee-cn/MuLUT)

# Usage

1. Train the network
```sh
cd sr

python 1_train_model.py \
    --stages 2 \
    --numSamplers 3 \
    --sampleSize 5 \
    --activateFunction gelu \
    -e ../models/autolut_mulut \
    --valDir ../data/SRBenchmark \
    --trainDir ../data/DIV2K \
    --no-lambda-lr
```

2. Transfer to LUTs

```sh
python 2_transfer_to_lut.py \
    --loadIter 200000 \
    -e ../models/autolut_mulut
```

3. Fine-tune LUTs

```sh
python 3_finetune_lut.py \
    --stages 2 \
    --numSamplers 3 \
    --sampleSize 5 \
    -e ../models/autolut_mulut/ \
    --no-lambda-lr \
    --valDir ../data/SRBenchmark \
    --trainDir ../data/DIV2K \
```

You'll see the PSNR values in the process of fine-tuning.

# Contact
Feel free to file an issue if you have any problems!
