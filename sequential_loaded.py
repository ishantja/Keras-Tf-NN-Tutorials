from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

new_model = load_model('models/medical_trial_model.h5')
new_model.summary()
print(new_model.get_weights())
print(new_model.optimizer)

# to import only architecture 
text_file = open("models/model_architecture.txt","r")
json_string = text_file.read()
model_architecture = model_from_json(json_string) 
model_architecture.summary()

# to import from YAML
# from tensorflow.keras.models import model_from_yaml
# model = model_from_yaml(yaml_string)

# to import only weights, we first need to create the same model 
model12 = Sequential([
    Dense(units=16, input_shape=(1,),activation='relu'), # (1,) is the shape of the manual input layer
    Dense(units=32,activation='relu'), # hidden layer with 32 cells
    Dense(units=2,activation='softmax') # output layer giving us probabilities for each output class
])

model12.load_weights('models/medical_trial_model_weights.h5')
print(model12.get_weights())