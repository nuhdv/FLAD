# FLAD
Journal of Computer Reasearch and Development - Anomaly Detection

This is the source code of the paper named **"Fusion Learning Based Unsupervised Anomaly Detection for Multi-Dimensional Time Series"** and published in Journal of Computer Reasearch and Development.

## Citation
Please cite our paper if you find this code is useful.  
Zhou Xiaohui, Wang Yijie, Xu Hongzuo, Liu Mingyu. Fusion Learning Based Unsupervised Anomaly Detection for Multi-Dimensional Time Series[J]. Journal of Computer Research and Development, 2023, 60(3): 496-508. doi: 10.7544/issn1000-1239.202220490.

## Usage
1. run main.py for sample usage. 
2. Data set: You may want to find the sample input data set in the "datasets" folder.
3. The input path can be an individual data set or just a folder.  
4. The performance might have slight differences between two independent runs. In our paper, we report the average auc with std over 5 runs. 


## Dependencies
```
Python 3.6
Troch == 1.7.0+cu110
pandas == 1.1.5
scikit-learn == 1.0.1
numpy == 1.21.6
```
