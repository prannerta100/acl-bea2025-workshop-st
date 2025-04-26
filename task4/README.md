#

```bash
# download data
wget https://raw.githubusercontent.com/kaushal0494/UnifyingAITutorEvaluation/refs/heads/main/BEA_Shared_Task_2025_Datasets/mrbench_v3_devset.json -O data/mrbench_v3_devset.json

wget https://raw.githubusercontent.com/kaushal0494/UnifyingAITutorEvaluation/refs/heads/main/BEA_Shared_Task_2025_Datasets/mrbench_v3_testset.json -O data/mrbench_v3_testset.json

# create virtual env and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# training data prep
python data.py

# train and save
python train.py
```
