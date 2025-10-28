### Data preparation:
- Or download the Synapse data following the Data preparation section of https://github.com/SLDGroup/EMCAD/tree/main

- Move train and test images into the root path: data_sets/Synapse/dataset13classes and naming train and test: `train_npz_new`,`test_vol_h5_new` respectively

### Training:
- Change either model V0 or V1 with the `model_name` parameter: `ResPVT_MambaCAU_V1` or `ResPVT_MambaCAU_V0`
- Run a file: experiments/btcv/train.py


### Testing:
- Verify the checkpoint path carefully before running test.py
- Run a file: experiments/btcv/test.py