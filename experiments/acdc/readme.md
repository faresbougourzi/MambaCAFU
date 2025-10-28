### Data preparation:
- download the ACDC data following the Data preparation section of https://github.com/SLDGroup/EMCAD/tree/main

- Move them into the root path: data_sets/ACDC

### Training:
- Change either model V0 or V1 with the `model_name` parameter: `ResPVT_MambaCAU_V1` or `ResPVT_MambaCAU_V0`
- Run a file: experiments/acdc/train.py


### Testing:
- Verify the checkpoint path carefully before running test.py
- Run a file: experiments/acdc/test.py
