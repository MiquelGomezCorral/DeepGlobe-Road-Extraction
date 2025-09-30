# DeepGlobe-Road-Extraction
RFA Proyect for DeepGlobe-Road-Extraction

# Dataset source
- link [here](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset)


# User project
- Create local env
```bash
 python3.12 -m venv venv
 source venv/bin/activate

 # install module
 pip install -e app/

 # install requirements
 pip install uv
 uv pip install -r requirements.txt


# For notebooks
pip install ipykernel
python -m ipykernel install --user --name=venv --display-name "Python (venv)"

# Pre-commit
sudo apt install pre-commit
pre-commit install
```
