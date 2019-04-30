# Dolar

Machine Learning timeseries prediction in Keras

# data comes from Datos.gob.ar

https://datos.gob.ar/dataset/sspm-tipo-cambio--usd---futuro-dolar

This is just an experiment it's impossible to predict long term values.

This neural network consists of a hybrid Convolution and a LSTM cell, the data is divided in sequences that are then feed into the model as a bunch of price sequences + the final dolar value as the "label".

Later the last values are feed to the model to predict the last series of labels given the last N predicted.
