### Data preparation:

- Download the ISIC17 data at https://challenge.isic-archive.com/data/#2017
- Move them (data_train.npy, mask_train.npy, data_val.npy, mask_val.npy, data_test.npy, test_mask.npy) into the root path: data_sets/ISIC2017/processed/

### Training:

- Change either model V0 or V1 with the `model_name` parameter: `ResPVT_MambaCAU_V1` or `ResPVT_MambaCAU_V0`
- Run a file: experiments/isic/train.py

### Testing:

- Verify the checkpoint path carefully before running test.py
- Run a file: experiments/isic/test.py
- Output file at test_log/<model_name>/test_result_0_bt_bestdsc.txt
