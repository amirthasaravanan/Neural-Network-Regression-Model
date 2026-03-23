# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## Theory

The objective of this project is to develop a Neural Network Regression Model that can accurately predict a target variable based on input features. The model will leverage deep learning techniques to learn intricate patterns from the dataset and provide reliable predictions.

## Neural Network Model

<img width="935" height="678" alt="image" src="https://github.com/user-attachments/assets/9b1986ca-aa16-48c6-a9d5-f71e7277bdd3" />

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: AMIRTHA VARSHINI M
### Register Number: 212224230017
```python
#Name: AMIRTHA VARSHINI M
#Register Number: 212224230017
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}
  def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the Model, Loss Function, and Optimizer
amirtha_brain=NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(amirtha_brain.parameters(), lr=0.001)

#Name: AMIRTHA VARSHINI M
#Register Number: 212224230017
def train_model(amirtha_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(amirtha_brain(X_train), y_train)
        loss.backward()
        optimizer.step()
        amirtha_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


```
## Dataset Information

<img width="236" height="410" alt="image" src="https://github.com/user-attachments/assets/c7dfc633-83e2-4b9a-9db5-bcbd3bb7228b" />


## OUTPUT
<img width="1729" height="267" alt="image" src="https://github.com/user-attachments/assets/37f1a400-a418-49de-9333-43d9c96ccbf8" />
<img width="1741" height="179" alt="image" src="https://github.com/user-attachments/assets/92a6365e-30ab-4255-b5eb-c8707d8046dc" />

## Training Loss Vs Iteration Plot
<img width="1738" height="687" alt="image" src="https://github.com/user-attachments/assets/6aa59485-a5e7-41ab-9621-6d37b9ada038" />

## New Sample Data Prediction
<img width="1737" height="142" alt="image" src="https://github.com/user-attachments/assets/4cfee976-695e-4e62-b3cd-69f94649bf97" />

## RESULT

The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
