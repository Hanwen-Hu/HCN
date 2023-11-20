# Iterative Reconstruction with Pattern Representation for Unsupervised Time Series Imputation

## Datasets
This repository contains the code of IR-Net+ and IR-GAIN+ with four datasets. Air Quality, Human Activity, Traffic Speed are available, but Solar Energy dataset should be firstly unzipped.

## Run
If you want to reproduce the experiment, please run "main.py" with Pycharm or with command
```angular2html
python3 main.py
```
The default dataset is Traffic Speed with 20% missing data. For the other cases, please add arguments such as
```angular2html
python3 main.py -dataset Solar -r_miss 0.4 -cuda_id 0 -plus True -iter_time 2
```
The argument `plus` determines whether to use the pattern representation layer, while `iter_time` represents the time of reconstruction, which are two main contributions of this work.

The argument `epochs` is set to 100, please do not reduce it, because there is no overfitting problem in IR-Net+, more training epochs will only result in better performance.

## Code
The code of IR-Net+ is in the package IR_Net_Plus, while that of IR-GAIN+ is in IR_GAIN_Plus. The pattern representation layer is written in IR_Net_Plus.

## Contact
If you have any questions or suggestions for our paper or codes, please contact us. Email: hanwen_hu@sjtu.edu.cn.