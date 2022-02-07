# Algorithmic_trading
**Introduction**: 
In the field of trading, after invention of Machine learning, we 
are using the tool of deep learning to develop a good strategy of trading. 
For the first trading day of each month we want our model to tell us if 
we are going to stay in the market for the current month or not.
Trading strategies are usually verified by backtesting 
reconstruct, with historical data, trades that would have occurred in 
the past using the rules that are defined with the strategy that you 
have developed
A time series is a sequence of numerical data points 
taken at successive equally spaced points in time. In investing, a 
time series tracks the movement of the chosen data points, such as 
the stock price, over a specified period of time with data points 
recorded at regular intervals.
Long short-term memory networks (LSTM) were first 
introduced in and have been used successfully in image and text 
classification tasks. At present, the exact structure of an LSTM layer is too 
long to describe and we refer the reader to [3] for details. Nonetheless, the 
key idea is that each layer consists of LSTM units, each of which has the 
ability to manage memory via a forget, input and output gate.


*Datasets*:
We used Yahoo Finance by import “Y finance” module 
form in Python to get the data of the S&P 500 index from the date 6 
month before today to today date. Our analysis is daily based.
We get the dataframe which contain the columns as:
High Low Open Close Adj.Close Volume
We use a data set available online that has time series data at 
one day intervals for all stocks in the S&P 500 between last 6 month before 
today to today date. However, in order to have a feasible strategy to act on, 
we only use timestamps. Each timestamp reports close price, high price, 
low price, open price and volume for 503 stocks. So finally we have data of 
all 503 companies.


**Methodology**: 
In the method section, we are using the LSTM(Long 
Short-Term Memory) networks which is a type of recurrent neural 
network(RNN) capable of learning order dependence in sequence 
prediction problems.
Long Short Term Memory (LSTM)
LSTM was proposed by Sepp Hochreiter and Jürgen 
Schmidhuber in 1997.
LSTMs are widely used for sequence prediction problems and have 
proven to be extremely effective. The reason they work so well is 
because LSTM is able to store past information that is important, and 

forget the information that is not. LSTM has three gates:
⦁ The input gate: The input gate adds information to the cell state
⦁ The forget gate: It removes the information that is no longer 
required by the model
⦁ The output gate: Output Gate at LSTM selects the information to 
be shown as output

⦁ 

After importing all required libraries and access the web data of stock 
from yahoo finance we define our LSTM model. 
Further steps of modeling: 
Data Normalisation: 
In the normalization of data first we use the MinMaxScaler.
MinMaxScaler which scales all the data features in the range 
[0, 1]. Now we use fit_transform() which is used on the training 
data to scale the training data and also learn the scaling parameters 
of that data. After that we separate as train and test. Training data is 
9/10 part of total data and rest is test data.
Incorporating Timesteps Into Data: 
We create two train variable as x_train and y_train & input our 
data in the form of a 3D array to the LSTM model. First, we create data in 
10 timesteps before using numpy to convert it into an array. Finally, we 

convert the data into a 3D array with X_train samples, 10 timestamps, and 
one feature at each step. And finally we create x_train , y_train, x_test& 
y_test. 
Creating the LSTM Model: 
We make a few imports from Keras: Sequential for initializing the 
neural network, LSTM to add the LSTM layer, Dropout for preventing 
overfitting with dropout layers, and Dense to add a densely connected 
neural network layer.
The LSTM layer is added with the following arguments: 50 units is the 
dimensionality of the output space,input_shape(10,1), 
return_sequences=True is necessary for stacking LSTM layers so the 
consequent LSTM layer has a 3-D sequence input, and input_shape is the 
shape of the training dataset. 
 We add the Dense layer that specifies an output of one unit. To 
compile our model we use the Adam optimizer and set the loss as the 
mean_squared_error. After that, we fit the model to run for 500 epochs 
(the epochs are the number of times the learning algorithm will work 
through the entire training set) with a batch size of 60 & verbose= 1.

Making Predictions on the Test Set: 
We modify the test set (notice similarities to the 
edits we made to the training set): merge the training set and the test set 
on the 0 axis, set 60 as the time step again, use MinMaxScaler, and reshape 
data. Then, inverse_transform puts the stock prices in a normal readable 
format.
Error Calculation:
Error calculation is a important part of any model. 
Despite being trained on the customized loss function, the models are 
evaluated on the basis of several metrics. The mean squared error (MSE), 
the accuracy in terms of predicted positive/negatives returns The MSE does 
not really give us any valuable information on how to utilize the model, but 
as the model is trained to minimize the customized MSE.We define the 
Mse_train and Mse_test. 
Define a funtion to get tickers of S&P 500:
We will get all tickers of S&P 500 from wikipedia by web scraping with help 

of BeautifulSoup module of Python. And then save these name as a file 
name S&P 500.pickle and then dump this file. Save all tickers in a list.We 
remove two comapanies from list name : "BRK.B" & "BF.B" because we get 
date error in these companies while calculating the closing price of these 
companies. Final list of the comapanies is of length 503. And we store all 
these companies in the dataframe name as final_dataframe. 
Calculate the closing price of each company:
Here we run a loop to get the closing price of each company by the help of 
Yfinance and store them into a list and add this list to final_dataframe
Predict the price of 30th day of each company:
Here we predict the price of 30th day by the help of our model that we 
created before, now we run a for loop for predict the price and store all 
predicted price into a list and add this list to final_dataframe

Calculate the predicted profit :
Here we divide predicted price of 30th day by closing price and get 
percentage return at 30th day of each company and store them in a list and 

add this list to the final_dataframe. Now we sort final_dataframe on the 
basis of predicted return, so that we can get all top companies that give us 
the maximum profit at 30th day and we take only top 20 companies. 
Calculate the no of stocks to buy: 
In this section we assume that our portfolio size is 20,000 , so we divide this 
value by 20. This is the price that will be invested in each company name as 
position_size. To get "no of share to buy" of each company, we divide 
postion_size with their corresponding share values.Thus create a new 
column in our final_dataframe.Then calculate closing price for each 
company.

**Result**: 

The result of our study focus on the accuracy of lstm model , 
that how much this model will predict the future stock price of any 
companyThe LSTM model can be tuned for various parameters such as 
changing the number of LSTM layers, adding dropout value or increasing 
the number of epochs. But are the predictions from LSTM enough to 
identify whether the stock price will increase or decrease


Future Recommendations :
The research and application of analysis and prediction 
methods on time series have a long history and rapid development, but the 
prediction effect of traditional methods fails to meet certain requirement 
of real application on some aspects in certain fields. Improving the 
prediction effect is the most direct research goal. Taking the study results 
of this paper as a starting point, we still have a lot of work in the future that 
needs further research. In the future, we will combine the EMD method or 
EEMD method with other methods, or use the LSTM method in 
combination with wavelet or VMD.
