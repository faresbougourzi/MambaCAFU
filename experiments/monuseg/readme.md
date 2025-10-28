### Data preparation:

- Download the MoNuSeg data at https://drive.google.com/drive/folders/1CsRmrOc7wwkPgJomQWMW32SU54xN2kMY?usp=sharing or https://github.com/McGregorWwww/UCTransNet/tree/main/datasets/MoNuSeg
- Move them (Train_Folder, Val_Folder, Test_Folder) into the root path: data_sets/MoNuSeg
- Run this script experiments/monuseg/create_Dataset_MoNuSeg.py to generate 5-fold.pt dataset

### Training:

- Change either model V0 or V1 with the `model_name` parameter: `ResPVT_MambaCAU_V1` or `ResPVT_MambaCAU_V0`
- Run a file: experiments/monuseg/train_test.py
- Output file at `outputs/model/MoNuSeg/<model_name>`/Results/mean.txt
