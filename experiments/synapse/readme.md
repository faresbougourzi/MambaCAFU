### Data preparation:
- Download the processed Synapse dataset at drive https://drive.google.com/file/d/1tGqMx-E4QZpSg2HQbVq5W3KSTHSG0hjK/view?usp=share_link . Or download the Synapse data following the Data preparation section of https://github.com/SLDGroup/EMCAD/tree/main

- Move them into the root path: data_sets/Synapse

### Training:
- Change either model V0 or V1 with the `model_name` parameter: `ResPVT_MambaCAU_V1` or `ResPVT_MambaCAU_V0`
- Run a file: experiments/synapse/train.py


### Testing:
- Verify the checkpoint path carefully before running test.py
- Run a file: experiments/synapse/test.py