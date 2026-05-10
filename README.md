# Waste Classification CNN

CNN-based garbage image classifier for 10-class waste categorisation.
Built with PyTorch for ENGR7761:MLCV Project 1.

By Anika Gardner gard0159 2163366

## Dataset
Garbage Classification Dataset (10 classes, 13,348 images)

## Project Structure
project/

├── main.py        

├── config.py      

├── model.py       

├── dataset.py     

├── trainer.py     

└── data/           #split data provided in ENGR7761 using split_data.py (run once) 

    ├── test.py     #70%

    ├── train.py    #15%
    
    ├── val.py      #15%
           


## Setup

```bash
# create and activate virtual environment
python -m venv venv
source venv/bin/activate          # mac/linux
venv\Scripts\activate             # windows

# install dependencies
pip install torch torchvision numpy pandas matplotlib kaggle
```

## Usage
```bash
python main.py #or python3
```
