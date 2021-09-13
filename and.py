from utils.model import Perceptron
from  utils.all_units import prepare_data


AND = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,0,0,1],
}
df = pd.DataFrame(AND)
X,y = prepare_data(df)
ETA = 0.3 # 0 and 1
EPOCHS = 10
model = Perceptron(eta=ETA, epochs=EPOCHS)
model.fit(X, y)
_ = model.total_loss()
save_model(model, "and.model")