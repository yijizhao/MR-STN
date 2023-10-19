# MRSTN #
This is an implementation of [Traffic inflow and outflow forecasting by modeling intra-and inter-relationship between flows (TITS, 2022)](https://ieeexplore.ieee.org/abstract/document/9827999/). MR-STN is a novel deep spatio-temporal network framework for traffic inflows and outflows forecasting. We show the generality and superiority of MR-STN by implementing it with four state-of-the-art graph-based deep spatio-temporal models, including STGCN, ASTGCN, STMGCN, and STSGCN.

## Requirements ##
- mxnet>=1.5.0
- easydict

Use ```nvcc -V``` to check the cuda version and install mxnet with the corresponding version. For example, use ```pip install mxnet-cu101``` to install mxnet for cuda version 10.1.

## Data ##
- Metro
- Taxi
- Highway
  
Please download the [data](https://pan.baidu.com/s/1CWG_5W-kEzZ81ZAVZM8eQw?pwd=8888) and unzip it in the ```./dataset``` directory.

## Usage ##
- python main.py --rid=1 --mode=stgcn --stack=3 --ed=4  --data=Metro
- python main.py --rid=1 --mode=astgcn --stack=3 --ed=4  --data=Metro
- python main.py --rid=1 --mode=stmgcn --stack=3 --ed=4  --data=Metro
- python main.py --rid=1 --mode=stsgcn --stack=3 --ed=4  --data=Metro

## Citing ##
If our paper benefits to your research, please cite our paper using the bitex below:

    @article{zhao2022traffic,
        title={Traffic inflow and outflow forecasting by modeling intra-and inter-relationship between flows},
        author={Zhao, Yiji and Lin, Youfang and Zhang, Yongkai and Wen, Haomin and Liu, Yunxiao and Wu, Hao and Wu, Zhihao and Zhang, Shuaichao and Wan, Huaiyu},
        journal={IEEE Transactions on Intelligent Transportation Systems},
        volume={23},
        number={11},
        pages={20202--20216},
        year={2022},
        publisher={IEEE}
    }
