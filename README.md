# 1. Update packages
sudo apt update

# 2. Install Python, pip, and virtual environment
sudo apt install python3 python3-pip python3-venv -y

# 3. Create a virtual environment
python3 -m venv dm-lab

# 4. Activate the environment
source dm-lab/bin/activate

# 5. Upgrade pip
python -m pip install --upgrade pip

# 6. Install Jupyter Notebook and all required libraries
pip install jupyter notebook pandas numpy matplotlib seaborn scikit-learn scipy openpyxl xlrd
