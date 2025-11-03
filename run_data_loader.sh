#Run data_loader script

#Activate virtual environment
source /home/Adam/Projects/MachineLearning/portfolio_ml/venv/bin/activate

#Run the script and append logs with timestamp
python /home/Adam/Projects/MachineLearning/portfolio_ml/src/data_loader.py >> /home/Adam/Projects/data_loader.log 2>&1
