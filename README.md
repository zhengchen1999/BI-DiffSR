# Binarized Diffusion Model for Image Super-Resolution

[Zheng Chen](https://zhengchen1999.github.io/), [Haotong Qin](https://htqin.github.io/), [Yong Guo](https://www.guoyongcs.com/), [Xiongfei Su](https://ieeexplore.ieee.org/author/37086348852), [Xin Yuan](https://en.westlake.edu.cn/faculty/xin-yuan.html), [Linghe Kong](https://www.cs.sjtu.edu.cn/~linghe.kong/), and [Yulun Zhang](http://yulunzhang.com/), "Binarized Diffusion Model for Image Super-Resolution", arXiv, 2024

[[arXiv]()] [visual results] [pretrained models]



#### 🔥🔥🔥 News

- **2024-06-09:** This repo is released.

---

> **Abstract:** Advanced diffusion models (DMs) perform impressively in image super-resolution (SR), but the high memory and computational costs hinder their deployment. Binarization, an ultra-compression algorithm, offers the potential for effectively accelerating DMs. Nonetheless, due to the model structure and the multi-step iterative attribute of DMs, existing binarization methods result in significant performance degradation. In this paper, we introduce a novel binarized diffusion model, BI-DiffSR, for image SR. First, for the model structure, we design a UNet architecture optimized for binarization. We propose the consistent-pixel-downsample (CP-Down) and consistent-pixel-upsample (CP-Up) to maintain dimension consistent and facilitate the full-precision information transfer. Meanwhile, we design the channel-shuffle-fusion (CS-Fusion) to enhance feature fusion in skip connection. Second, for the activation difference across timestep, we design the timestep-aware redistribution (TaR) and activation function (TaA). The TaR and TaA dynamically adjust the distribution of activations based on different timesteps, improving the flexibility and representation alability of the binarized module. Comprehensive experiments demonstrate that our BI-DiffSR outperforms existing binarization methods.

![](figs/BI-DiffSR.png)

---

---

|                            HR                             |                              LR                              | [SR3 (FP)](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement) |          [BBCU](https://github.com/Zj-BinXia/BBCU)          |                       BI-DiffSR (ours)                       |
| :-------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :---------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="figs/compare/ComS_img_023_HR_x4.png" height=80> | <img src="figs/compare/ComS_img_023_Bicubic_x4.png" height=80> |  <img src="figs/compare/ComS_img_023_SR3_x4.png" height=80>  | <img src="figs/compare/ComS_img_023_BBCU_x4.png" height=80> | <img src="figs/compare/ComS_img_023_BI-DiffSR_x4.png" height=80> |
| <img src="figs/compare/ComS_img_033_HR_x4.png" height=80> | <img src="figs/compare/ComS_img_033_Bicubic_x4.png" height=80> |  <img src="figs/compare/ComS_img_033_SR3_x4.png" height=80>  | <img src="figs/compare/ComS_img_033_BBCU_x4.png" height=80> | <img src="figs/compare/ComS_img_033_BI-DiffSR_x4.png" height=80> |

## TODO

* [ ] Release code and pretrained models

## Contents

1. Datasets
1. Models
1. Training
1. Testing
1. [Results](#results)
1. [Citation](#citation)
1. [Acknowledgements](#acknowledgements)

## <a name="results"></a> Results

We achieved state-of-the-art performance. Detailed results can be found in the paper.

<details>
<summary>Click to expand</summary>




- quantitative comparisons in Table 2 (main paper)

<p align="center">
  <img width="900" src="figs/T1.png">
</p>



- visual comparison in Figure 8 (main paper)

<p align="center">
  <img width="900" src="figs/F1.png">
</p>



- visual comparison in Figure 13 (supplemental material)

<p align="center">
  <img width="900" src="figs/F2-1.png">
  <img width="900" src="figs/F2-2.png">
</p>




- visual comparison in Figure 14 (supplemental material)

<p align="center">
  <img width="900" src="figs/F3-1.png">
  <img width="900" src="figs/F3-2.png">
</p>

</details>



## <a name="acknowledgements"></a> Acknowledgements

This code is built on [BasicSR](https://github.com/XPixelGroup/BasicSR), [Image-Super-Resolution-via-Iterative-Refinement](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement).
