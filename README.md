# AnyTSR: Any-Scale Thermal Super-Resolution for UAV 

### MengyuanLi, Changhong Fu*, Ziyu Lu, Zijie Zhang, Haobo Zuo, Liangliang Yao
\* Corresponding author.

## Abstract
Thermal imaging can greatly enhance the application of intelligent unmanned aerial vehicles (UAV) in challenging environments. However, the inherent low resolution of thermal sensors leads to insufficient details and blurred boundaries. Super-resolution (SR) offers a promising solution to address this issue, while most existing SR methods are designed for fixed-scale SR. They are computationally expensive and inflexible in practical applications. To address above issues, this work proposes a novel any-scale thermal SR method (AnyTSR) for UAV within a single model. Specifically, a new image encoder is proposed to explicitly assign specific feature code to enable more accurate and flexible representation. Additionally, by effectively embedding coordinate offset information into the local feature ensemble, an innovative any-scale upsampler is proposed to better understand spatial relationships and reduce artifacts. Moreover, a novel dataset (UAV-TSR), covering both land and water scenes, is constructed for thermal SR tasks. Experimental results demonstrate that the proposed method consistently outperforms state-of-the-art methods across all scaling factors as well as generates more accurate and detailed high-resolution images.
![Workflow of our AnyTSR](https://github.com/MengyuanLi1106/AnyTSR/blob/main/image/figure.png)

This figure shows the workflow of our AnyTSR.

## 🔥 New Extension: AnyTSR++
We further extend AnyTSR and propose a more advanced framework:
# **AnyTSR++: Prompt-Oriented Any-Scale Thermal Super-Resolution for Unmanned Aerial Vehicle**

📌 **Dataset and Code**

👉 https://github.com/vision4robotics/AnyTSRpp

If you are interested in our extended work, please refer to the above repository.

## About Code
### 1. Environment setup
This code has been tested on Ubuntu 22.04, Python 3.10.15, Pytorch 1.13.0, CUDA 11.7.
Please install related libraries before running this code: 
```bash
pip install -r requirements.txt
```

### 2. Test

```bash 
python demo.py                                
```
The testing result will be saved in the `output` directory.

### 3. Contact
If you have any questions, please contact me.

Mengyuan Li

Email: [mengyuanli@tongji.edu.cn](mengyuanli@tongji.edu.cn)

For more evaluations, please refer to our paper.

## References 

```

```
