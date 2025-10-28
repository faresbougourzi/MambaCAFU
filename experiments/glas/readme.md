### Data preparation:

- Download the GlaS data at https://drive.google.com/drive/folders/1CsRmrOc7wwkPgJomQWMW32SU54xN2kMY?usp=sharing OR  https://drive.google.com/file/d/1t2MDLkj5DYyGYBGeT7Q7u8F6rscqNzCF/view
- Move them (xxximg.bmp and xxxmask.bmp) into the root path: data_sets/GlaS
- Run this script experiments/glas/create_Dataset_GlaS.py to generate 5-fold.pt dataset

### Training:

- Change either model V0 or V1 with the `model_name` parameter: `ResPVT_MambaCAU_V1` or `ResPVT_MambaCAU_V0`
- Run a file: experiments/glas/train_test.py
- Output file at `outputs/model/GlaS/<model_name>`/Results/mean.txt
