# Waste Classification CNN

CNN-based garbage image classifier for 10-class waste categorisation.
Built with PyTorch for ENGR7761:MLCV Project 1.

By Anika Gardner gard0159 2163366

## Dataset
Garbage Classification Dataset (10 classes, 13,348 images)
- Classes: battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, trash
- Source: https://www.kaggle.com/datasets/mostafaabla/garbage-classification

### Kaggle Setup
1. Get Kaggle API key from kaggle.com/settings
2. Place `kaggle.json` in `~/.kaggle/`
3. Dataset downloads automatically on first run

## Project Structure
project/

├── main.py        

├── config.py      

├── model.py       

├── dataset.py     

├── trainer.py     

└── data/         


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
python main.py
```
