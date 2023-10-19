# MRSTN #
The official implementation of 'Traffic inflow and outflow forecasting by modeling intra-and inter-relationship between flows' (TITS, 2022). MR-STN is a novel deep spatio-temporal network framework for traffic inflows and outflows forecasting. We show the generality and superiority of MR-STN by implementing it with four state-of-the-art graph-based deep spatio-temporal models, including STGCN, ASTGCN, STMGCN, and STSGCN.

## Requirements ##
- mxnet>=1.5.0
- easydict

## Data ##
- Metro
- Taxi
- Highway

## Usage ##
python main.py --rid=1 --mode=stgcn --stack=3 --ed=4  --data=Metro
python main.py --rid=1 --mode=astgcn --stack=3 --ed=4  --data=Metro
python main.py --rid=1 --mode=stmgcn --stack=3 --ed=4  --data=Metro
python main.py --rid=1 --mode=stsgcn --stack=3 --ed=4  --data=Metro

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