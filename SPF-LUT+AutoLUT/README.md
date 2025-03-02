# \[CVPR 2024\] Look-Up Table Compression for Efficient Image Restoration

Yinglong Li, [Jiacheng Li](https://ddlee-cn.github.io/), [Zhiwei Xiong](http://staff.ustc.edu.cn/~zwxiong/)

[CVPR Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Look-Up_Table_Compression_for_Efficient_Image_Restoration_CVPR_2024_paper.html)

![Overview of DFC](https://github.com/leenas233/DFC/blob/main/docs/DFC_overview.png)

## Usage
Updating! Any questions, please contact me at any time.
### Dataset
| task             | training dataset                                      | testing dataset                                                                                                                               |
| ---------------- | ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| super-resolution | [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)    | Set5, Set14, [BSDS100](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/), Urban100, [Manga109](http://www.manga109.org/en/)   |
| denoising        | DIV2K                                                 | Set12, BSD68                                                                                                                                  |
| deblocking       | DIV2K                                                 | [Classic5](https://github.com/cszn/DnCNN/tree/master/testsets/classic5), [LIVE1](https://live.ece.utexas.edu/research/quality/subjective.htm) |
| deblurring       | [GoPro](https://seungjunnah.github.io/Datasets/gopro) | GoPro test set                                                                                                                                |
### Pretrained Models
Some pretrained LUTs and their compressed version can be download [here](https://drive.google.com/drive/folders/1nxPzhpLdZut-16T_Z3b-5Oo9uU4Dbe1h?usp=drive_link).
### Step 1: Training LUT network
Let's take the SPF-LUT for x4 sr as an example.
```shell
cd sr
python 1_train_model.py --model SPF_LUT_net --scale 4 --modes sdy --expDir ../models/spf_lut_x4 --trainDir ../data/DIV2K --valDir ../data/SRBenchmark
```
The trained LUT network will be available under the `../models/spf_lut_x4` directory.
### Step 2: Transferring LUT network into compressed LUTs
```shell
python .\2_compress_lut_from_net.py --model SPF_LUT_net --scale 4 --modes sdy --expDir ../models/spf_lut_x4 --lutName spf_lut_x4 --cd xyzt --dw 2 --si 5
```
The compressed LUTs will be available under the `../models/spf_lut_x4` directory. `--cd`: The number of compressed dimensions; `--dw`: Diagonal width; `--si`: Sampling interval of non-diagonal subsampling.
### Step 3: Fine-tuning compressed LUTs
```shell
python 3_finetune_lut.py --model SPF_LUT_DFC --scale 4 --modes sdy --expDir ../models/spf_lut_x4  --trainDir ../data/DIV2K --valDir ../data/SRBenchmark --load_lutName spf_lut_x4 --cd xyzt --dw 2 --si 5
```
The finetuned compressed LUTs will be available under the `../models/spf_lut_x4` directory.
### Step 4: Test compressed LUTs
```shell
python .\4_test_SPF-LUT_DFC.py --scale 4 --modes sdy --expDir ../models/spf_lut_x4 --testDir ../data/SRBenchmark --lutName weight --cd xyzt --dw 2 --si 5
```
### Contact
If you have any questions, feel free to contact me any time by e-mail `yllee@mail.ustc.edu.cn`
## Citation
If you found our implementation useful, please consider citing our paper:
```bibtex
@InProceedings{Li_2024_CVPR, 
	author = {Li, Yinglong and Li, Jiacheng and Xiong, Zhiwei}, 
	title = {Look-Up Table Compression for Efficient Image Restoration}, 
	booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
	month = {June}, year = {2024}, pages = {26016-26025} 
}
```

## Acknowledgement
This work is based on the following works, thank the authors a lot.

[SR-LUT](https://github.com/yhjo09/SR-LUT)

[MuLUT](https://github.com/ddlee-cn/MuLUT/tree/main)