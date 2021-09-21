from utils.models import Perceptron
from utils.all_utils import prepare_data
from utils.all_utils import save_model
import pandas as pd
import logging
import os


logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
logs_dir = 'logs'
os.makedirs(logs_dir, exist_ok=True)
logging.basicConfig(filename = os.path.join(logs_dir, 'AND_Logs.log'), level=logging.INFO, format=logging_str,filemode='a')

def main(data,eta,epochs,filename):

    df = pd.DataFrame(AND)
    logging.info(f"This is a actual dataframe{df}")
    X,y = prepare_data(df)

    model = Perceptron(eta, epochs)
    model.fit(X, y)
    _ = model.total_loss()
    save_model(model, filename)

if __name__ == '__main__':
    AND = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,0,0,1],
    }

    ETA = 0.3 # 0 and 1
    EPOCHS = 100
    filename = "and.model"
    try:
        logging.info("\n <<<<<<<  Training started sucessfully >>>>>>> \n")
        main(data=AND,eta=ETA,epochs=EPOCHS,filename=filename)
        logging.info("\n <<<<<<<  Training ended sucessfully >>>>>>> \n")
    except Exception as e:
        logging.exception(e)
        raise e