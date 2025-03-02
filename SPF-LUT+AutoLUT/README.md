# AutoLUT applied on SPF-LUT

Original SPF-LUT: [repo](https://github.com/leenas233/DFC)

# Usage
## Dataset
| task             | training dataset                                      | testing dataset                                                                                                                               |
| ---------------- | ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| super-resolution | [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)    | Set5, Set14, [BSDS100](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/), Urban100, [Manga109](http://www.manga109.org/en/)   |
| denoising        | DIV2K                                                 | Set12, BSD68                                                                                                                                  |
| deblocking       | DIV2K                                                 | [Classic5](https://github.com/cszn/DnCNN/tree/master/testsets/classic5), [LIVE1](https://live.ece.utexas.edu/research/quality/subjective.htm) |
| deblurring       | [GoPro](https://seungjunnah.github.io/Datasets/gopro) | GoPro test set                                                                                                                                |

1. Train the network
```sh
cd sr
python 1_train_model.py \
    --model SPF_LUT_net \
    --scale 4 \
    --modes s \
    --expDir ../models/spf_lut_autolut \
    --trainDir ../data/DIV2K \
    --valDir ../data/SRBenchmark \
    --sample-size 5
```

Node the `--modes s` part. We're using AutoSample, so the modes `s`, `d` and `y` doesn't actually mean anything,
we're just creating one branch for each character. So it would be same if you wrote `--modes a`.

The above command is for SPF-Light. If you want three branches, use `--modes sdy`.

2. Export to LUT

```sh
python 2_compress_lut_from_net.py \
    --expDir ../models/spf_lut_autolut/ \
    --scale 4 \
    --modes s \
    --sample-size 5 \
    --loadIter 200000 \
    --lutName spf_lut_x4
```

3. Finetune LUTs

```sh
python 3_finetune_compress_lut.py \
    --model SPF_LUT_DFC \
    --scale 4 \
    --modes s \
    --expDir ../models/spf_lut_autolut/ \
    --trainDir ../data/DIV2K \
    --valDir ../data/SRBenchmark \
    --sample-size 5
```

You'll be able to see the PSNR values in the process of finetuning.
