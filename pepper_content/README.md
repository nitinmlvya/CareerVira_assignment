## Python Version
- 3.7.9

## Install python packages
pip install -r requirements.txt


## Train model:
- Goto parent directory (pepper_content)
- run command `python train_model.py`

## Predict top 3 user_ids for given assignments
- Goto parent directory (pepper_content)
- run command 

`python3 predict_result.py --input "data/assignments.xlsx" --output "assignments_result.csv"`

Here, 
- **--input = Input file path**, Excel sheet of assignments.xlsx. It must be in the same format which you had shared me during assignment/use case.
- **--output = Output file path**, It stores top 3 user IDS for their respective assignments.
