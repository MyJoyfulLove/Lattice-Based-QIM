# Content-Aware Quantization Index Modulation

**Authors:** Junlong Mao, Huiyi Tang, Shanxiang Lyu, Zhengchun Zhou, Xiaochun Cao

## Abstract
QIM is a convenient and efficient data hiding paradigm. However, QIM does not take the distribution of carriers and watermarks into consideration, which leads to suboptimal results when dealing with non-uniformly distributed carriers and watermarks. We propose a lattice-based CA-QIM method to reduce the distortion of this series of methods. The specific operation is to adaptively modify the lattice coset and the corresponding codebook by counting the correspondence between the carrier and the secret information. At the same time, we considered the robustness of QIM and quantized the carrier to the correct decoding area (i.e., Voronoi Cell) instead of the lattice point, further reducing the distortion.

## Codes
* MD-QIM: Minumum Distortion QIM, reproduced from [Lattice-Based Minimum-Distortion Data Hiding](https://ieeexplore.ieee.org/abstract/document/9455352)
* CA-QIM: Content-Aware QIM, also combined with MD-QIM
* SDCVP: Lattice Encode and Decode

## Citation
J. Mao, H. Tang, S. Lyu, Z. Zhou and X. Cao, "[Content-Aware Quantization Index Modulation: Leveraging Data Statistics for Enhanced Image Watermarking](https://ieeexplore.ieee.org/abstract/document/10356112)," in IEEE Transactions on Information Forensics and Security, vol. 19, pp. 1935-1947, 2024, doi: 10.1109/TIFS.2023.3342612.
