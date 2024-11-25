#!/bin/bash

# 切换到目标目录
cd /home/dongsheng/Code/ECNU/TCMEval/CAT

# 激活conda环境
source /home/dongsheng/anaconda3/bin/activate
conda activate sds

# 设置PYTHONPATH环境变量
export PYTHONPATH=/home/dongsheng/Code/ECNU/TCMEval/

# 运行第一个Python脚本
python CAT_train_irt.py

# 运行第二个Python脚本
python CAT_procedure.py
