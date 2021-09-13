from utils.models import Perceptron
from utils.all_utils import prepare_data
from utils.all_utils import save_model
import pandas as pd

def main(data,eta,epochs,filename):

    df = pd.DataFrame(AND)
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
    EPOCHS = 10
    filename = "and.model"

    main(data=AND,eta=ETA,epochs=EPOCHS,filename=filename)