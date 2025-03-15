# [CVPR 2025] AutoLUT: LUT-Based Image Super-Resolution with Automatic Sampling and Adaptive Residual Learning

This is the official repo for CVPR 2025 paper "AutoLUT: LUT-Based Image Super-Resolution with Automatic Sampling and Adaptive Residual Learning".

---

Authors: Yuheng Xu, Shijie Yang, Xin Liu, Jie Liu, Jie Tang, Gangshan Wu

[paper](https://arxiv.org/abs/2503.01565)

# Usage

1. Build the environment:
```sh
conda env create -f AutoLUTEnvironment.yaml
```

2. Follow the README.md in each folder.

# SR Results

You can download the super-resolution results from [aliyun drive](https://www.alipan.com/s/4LSqy2TF1mg)

Because Aliyun drive doesn't support sharing `.tar.gz` files, I encoded them with `base64` and uploaded them as text files. You'll need to extract them like this:

```console
cat file.txt | base64 -d | tar -xzvf -
```

# Contact

If you have any problems, please feel free to file an issue.

# Star History
[![Star History Chart](https://api.star-history.com/svg?repos=SuperKenVery/AutoLUT&type=Date)](https://star-history.com/#SuperKenVery/AutoLUT&Date)

# Acknowledgments

- [SR-LUT](https://github.com/yhjo09/SR-LUT)
- [MuLUT](https://github.com/ddlee-cn/MuLUT/tree/main)
- [SPF-LUT](https://github.com/leenas233/DFC)
