# Introduction
This repository is provided for replicating the canonical recognition methods of SSVEP signals. The replicated methods include:

- <a href="https://blog.csdn.net/weixin_43715601/article/details/120567547">CCA (Canonical Correlation Analysis)</a> [1] 

- <a href="https://blog.csdn.net/weixin_43715601/article/details/130019520">MSI (Multivariate Synchronization Index)</a> [2]

- <a href="https://blog.csdn.net/weixin_43715601/article/details/120576642">FBCCA (Filter Bank Canonical Correlation Analysis)</a> [3]

- <a href="https://blog.csdn.net/weixin_43715601/article/details/120575420 ">TRCA (Task-Related Component Analysis)</a> [4]

- <a href="https://blog.csdn.net/weixin_43715601/article/details/144363774">TDCA (Task-Discriminant Component Analysis)</a> [5]

The file distribution follow the code desgin of <a href="https://github.com/YuDongPan/SSVEPNet">SSVEPNet</a> [6]. And a 12-class public dataset [7] was used to conduct evaluation.


# Running Environment
* Setup a virtual environment with python 3.8 or newer
* Install requirements

```
pip install -r Resource/requirements.txt
```

## Running Demo Experiments
```
cd Exeperiment
python TDCA_SSVEP_Classification.py
```

# Reference
[1] Lin Z, Zhang C, Wu W, et al. Frequency recognition based on canonical correlation analysis for SSVEP-based BCIs[J]. IEEE transactions on biomedical engineering, 2006, 53(12): 2610-2614. <a href="https://ieeexplore.ieee.org/abstract/document/4015614">https://ieeexplore.ieee.org/abstract/document/4015614</a>

[2] Zhang Y, Xu P, Cheng K, et al. Multivariate synchronization index for frequency recognition of SSVEP-based brain–computer interface[J]. Journal of neuroscience methods, 2014, 221: 32-40. <a href="https://www.sciencedirect.com/science/article/abs/pii/S0165027013002677">https://www.sciencedirect.com/science/article/abs/pii/S0165027013002677</a>

[3] Chen X, Wang Y, Gao S, et al. Filter bank canonical correlation analysis for implementing a high-speed SSVEP-based brain–computer interface[J]. Journal of neural engineering, 2015, 12(4): 046008. <a href="https://iopscience.iop.org/article/10.1088/1741-2560/12/4/046008/meta">https://iopscience.iop.org/article/10.1088/1741-2560/12/4/046008/meta</a>

[4] Nakanishi M, Wang Y, Chen X, et al. Enhancing detection of SSVEPs for a high-speed brain speller using task-related component analysis[J]. IEEE Transactions on Biomedical Engineering, 2017, 65(1): 104-112. <a href="https://ieeexplore.ieee.org/abstract/document/7904641">https://ieeexplore.ieee.org/abstract/document/7904641</a>

[5] Liu B, Chen X, Shi N, et al. Improving the performance of individually calibrated SSVEP-BCI by task-discriminant component analysis[J]. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2021, 29: 1998-2007. <a href="https://ieeexplore.ieee.org/abstract/document/9541393">https://ieeexplore.ieee.org/abstract/document/9541393</a>

[6] Pan Y, Chen J, Zhang Y, et al. An efficient CNN-LSTM network with spectral normalization and label smoothing technologies for SSVEP frequency recognition[J]. Journal of Neural Engineering, 2022, 19(5): 056014. <a href="https://iopscience.iop.org/article/10.1088/1741-2552/ac8dc5/meta">https://iopscience.iop.org/article/10.1088/1741-2552/ac8dc5/meta</a>

[7] Nakanishi M, Wang Y, Wang Y T, et al. A comparison study of canonical correlation analysis based methods for detecting steady-state visual evoked potentials[J]. PloS one, 2015, 10(10): e0140703. <a href="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0140703">https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0140703</a>

