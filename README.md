# Iterative Time Series Imputation by Maintaining Dependency Consistency

## Datasets
This repository contains the code of IR-Square-Net and IR-Square-GAIN with four datasets. Air Quality, Human Activity, Traffic Speed are available, but Solar Energy dataset should be firstly unzipped.

## Run
If you want to reproduce the experiment, please run "main.py" with Pycharm or with command
```angular2html
python3 main.py
```
The default dataset is Air Quality with 20% missing data. For the other cases, please add arguments such as
```angular2html
python3 main.py -model GAN -dataset traffic -r_miss 0.4 -cuda_id 0 -use_irm 1 -iter_time 2
```
The argument `use_irm` determines whether to use the incomplete representation mechanism, while `iter_time` represents the time of reconstruction, which are two main contributions of this work.

## Contact
If you have any questions or suggestions for our paper or codes, please contact us. Email: hanwen_hu@sjtu.edu.cn.