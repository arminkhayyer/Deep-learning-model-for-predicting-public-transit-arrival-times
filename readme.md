# Deep learning model for predicting public transit arrival times 


This is the implementation of the "Deep learning model for predicting public 
transit arrival times" paper.
The methodology combines an Artificial Neural Network approach that makes 
predictions based on static featuresand a Recurrent Neural Network which
captures time series patterns. We use a New Orleans Transit Authority (NORTA) dataset 
collected by Automatic Passenger Counter (APC) devices to calibrate and test our method of forecasting for both buses and streetcars. 
The results of ourexperiments show modest improvement in terms of mean absolute percentage of error compared to existing models.

A sample of the dataset containing 100 data rows is provided in the "Data" folder. In order to run the code, first install the required packages using the following command, some of the required libraries are Tensorflow, keras, SickitLearn, and so on.  

```bash
pip install -r requirements.txt
```
Once the required packages are installed, then the "code.py" file can be run. This will first cleans the data, preprosess the data, construct the models, and finally trains the models over the training sets, prints and plots the results over the test set. 
## Usage
The main function is as follows: 

```python
filename = "Example.xlsx"
    look_back = 30
    verbose = 2
    epochs = 200
    minibatch = 512
    learning_rate = 0.01
    N_points_to_plot = 200
    route = Route(filename, look_back, epochs, minibatch, verbose, learning_rate, N_points_to_plot)
    route.Run()
    route.Print_results()
    route.Plot()
```
While most of the variables are self-explanatory, the N-points_to_plot variable shows the number of rows of the test set that the user wants to plot its results. 

## Authors

Armin Khayyer

![alt text][6.1] : https://github.com/arminkhayyer/Deep-learning-model-for-predicting-public-transit-arrival-times

[6.1]: http://i.imgur.com/0o48UoR.png (github icon with padding)


![iconfinder_circle-linkedin_317750](https://user-images.githubusercontent.com/22824676/87214685-953f8c00-c2f4-11ea-91a7-46afd6c3b197.png)
https://www.linkedin.com/in/armin-khayyer/
## More Details
Users are reffered to the paper for more details. 
