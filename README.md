# deep-learning-challenge


## Project Structure
```models/```: Contains the saved models in HDF5 format.

    - AlphabetSoupCharity_Optimization.h5: The optimized model.
    - AlphabetSoupCharity.h5: The initial model.
```notebooks/```: Contains Jupyter notebooks for data analysis and model training.

    - AlphabetSoupCharity_Optimized.ipynb: Notebook for the optimized model.
    - AlphabetSoupCharity.ipynb: Notebook for the initial model.

## Execution

To run the code and train the models, you will need an environment with sufficient resources. We used Google Colab, a free online tool that can host Python notebooks and utilizes Google's resources.

Once you have an appropriate environment, execute one of the notebooks based on the model you wish to explore. Please note that training the models requires some time. The initial model with lower complexity may take 5-10 minutes, while more advanced configurations may take 20-30 minutes to complete training.

## Analysis 

This project was developed to help Alphabet Soup create a tool for selecting applicants for funding who have the highest likelihood of success in their ventures. By utilizing machine learning and neural networks, we leveraged the features in the provided dataset to develop a binary classifier that can predict the success of applicants funded by Alphabet Soup.

### Preprocessing

Our first step was to clean and pre-process the data. This included the following steps in order: 

1. Drop non-beneficial ID columns, 'EIN' and 'NAME'.
2. Identified 'noise' in the dataset for the ```APPLICATION_TYPE``` and ```CLASSIFICATION``` columns. Values with a total count less than 528 and 1883 respectively were grouped into a bucket labeled 'other'.
3. Convert categorical data to numeric with ```pd.get_dummies()```
4. Split our data into a training and testing dataset using ```sklearn.model_selection.train_test_split()```
5. Used sklearn's StandardScaler preprocessor in order to scale the data.

### Training the Model

For our first model, we followed the following steps to compile and train our model. 

1. Defined the amount of input features in our dataset. 
2. Created a ```tf.keras.models.Sequential()``` model.
3. Added 3 layers. An input layer, second hidden layer, and finally, our output layer. 

### Results and Optimizations

The first model we created achieved an accuracy score of 0.73, with a loss of 0.55.

After creating our initial model, some optimization choices were made in order to gain improvement in our accuracy and loss scores. 

#### Manual Tuning

We first attempted the following:

1. Add 2 additional hidden layers for processing
2. Increased neuron count in each layer. 
3. Added a dropout layer to prevent overfitting

The changes above results in an accuracy score of 0.74 and a loss of 0.57. 

#### keras-tuner

After manual tuning, we then turned to keras-tuner. We will use this to automate hyper-parameter configuration testing.

1. Allowed our keras-tuner to choose between relu, tanh, and sigmoid activation functions.
2. Allowed our keras-tuner to choose between 1 and 10 neurons for the input layer.
3. Created a loop to create 5 hidden layers with varying activation functions and neuron counts in each one.
4. Compiled the model and allowed our keras-tuner to iterate through 60 trials to identify the best fit for hyper parameters.

After using keras-tuner to automate hyper-parameter configuration, we identified our best model with an accuracy of 0.74 and a loss of 0.56.

## Summary

After testing multiple configurations of models on this dataset, we observed that all models performed similarly, with about a ~1% point variance in final accuracy and loss results. This suggests that while our current model performs well, exploring different types of models could potentially yield higher accuracy scores.

Overall, this project demonstrates the potential of machine learning in aiding decision-making processes for funding applicants, providing a valuable tool for Alphabet Soup to maximize the success of their funded ventures.




