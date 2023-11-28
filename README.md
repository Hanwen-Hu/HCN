# An Iterative Time Series Imputation Network by Maintaining Pattern Consistency

## Datasets
This repository contains the code of IR-Square-Net and IR-Square-GAIN with four datasets. Air Quality, Human Activity, Traffic Speed are available, but Solar Energy dataset should be firstly unzipped.

## Run
If you want to reproduce the experiment, please run "main.py" with Pycharm or with command
```angular2html
python3 main.py
```
The default dataset is Traffic Speed with 20% missing data. For the other cases, please add arguments such as
```angular2html
python3 main.py -model GAIN -dataset Solar -r_miss 0.4 -cuda_id 0 -irm_usage True -iter_time 2
```
The argument `irm_usage` determines whether to use the incomplete representation mechanism, while `iter_time` represents the time of reconstruction, which are two main contributions of this work.

The argument `epochs` is set to 50, please do not reduce it, because there is no overfitting problem in IR-Net+, more training epochs will only result in better performance.

## Code
Package `IR_Square_Net` and `IR_Square_GAIN` contain the codes of corresponding models. The incomplete representation mechanism `IRM` lies in IR_Square_Net.

## Contact
If you have any questions or suggestions for our paper or codes, please contact us. Email: hanwen_hu@sjtu.edu.cn.