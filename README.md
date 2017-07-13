This is the practice of entity recognization based on context-BoW + LR/DNN/CNN methods.
Please read the notes.pdf under 'notes' folder for more details.

### Test Data Result
You can find the core terms and brand prediction for test data at 'result/test_data_result.txt'.

### Dependency
sudo pip install -U scikit-learn
sudo pip install theano
sudo pip install keras

### Model Download
Please download the model files and put these files into 'model' folder before you run model prediction program.

core_term_dnn.model: https://drive.google.com/open?id=0B3mPA5jcECZTa0VQRXBwM2RmU3c
brand_dnn.model: https://drive.google.com/open?id=0B3mPA5jcECZTLTA1YTNMbWtqNjA

### Model Prediction
```
cd entity_recognization
# Turn on the following statement when you want to train model with gpu.
# export THEANO_FLAGS=device=gpu, floatX=float32
pytho forecast.py
```

### Model Training
```
cd entity_recognization
# The default model is 3-gram BoW with deep neural networks.
python train.py
```
