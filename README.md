# Mortgage Market Tri Analysis

Optimizing Capital Allocation for Mortgage Market Loans  
Naman Jain, February 2017

## Contents:
- Motivation
- For each of the 3 goals:
	* Data
	* Approach
	* Results
- Next Steps

## Motivation:
This project was developed as a capstone project for Galvanize's Data Science 	 program.

I worked with data from [Shubham Housing Finance](http://www.shubham.co/), a firm in India that has given out more than USD $150 Million as mortgage loans over the past 5 years.  

My goal was to use data science to help the firm optimize its usage of capital, both in its loan allocation process and in its expansion.  

I decided to break this broad goal down into 3 individual more specific goals:  
1) Build a classifier that predicts the probability that a customer will default on their loan  
2) Recommend new office locations which maximize growth potential  
3) [Forecast upcoming amount of business over the next quarter](https://github.com/Namanj/Mortgage-Market-Tri-Analysis/blob/master/README.md## Forecasting Business:)  

Want a quick 4 minute overview? I gave a lightening presentation which can be found here:
<p align="center">
  <a href="https://www.youtube.com/watch?v=F0mVX1ReX2s">
  <img src='/images/Presentation.png'>
</a>
</p>



## Loan Default Classifier:
First of my goals was to build a classifier that can predict the probability of a customer defaulting on their loan payment. If the firm can recognize such customers ahead of time then it can do a better job of getting ahead of the problem.

This problem was harder than it seems for 2 reasons:  
1) My target audience are people who have been excluded from the formal economy and hence don’t have the means to demonstrate their fiscal responsibility over time. As a result, I had to rely upon feature engineering in order to evaluate default probability amongst them  
2) Only 3% of the people ever default on their loans, making the classes very imbalanced  

### Data
I used data provided to me by the firm which consisted of all the loans that they had given over the past 5 years, along with their performance.  
There were a total of ~15k data points with each representing a loan, and there were 40 columns of which 20 were categorical, 19 were numerical and 1 was temporal.

### Approach
I spent quite some time reducing the dimensionality of my data by sub-selecting the categorical variables based on signal to noise ratio, sub-selecting the numerical columns by thinking about how they would correlate with my dependent variable, and by doing some feature engineering.  

I eventually decided to use these features in my model:
- Type - Whether loan has been downsized
- Product Par Name - Loan product category
- Name of Proj - Whether loan was processed as part of a Housing project, or if it was given directly to the customer
- Rl/ Urb - Whether customer is in a Rural or Urban location
- EMI - Easy Monthly installment, the amount which customer has to pay every month
- ROI - Represents rate of interest charged to client for the product provided
- Tenor - Represent loan repayment period
- Amt Db - Represents amount of loan disbursed to client till Nov '16

I decided to use the AdaBoost ensemble model, which I tuned using a GridSearch. The code can be found  
[here](https://github.com/Namanj/Mortgage-Market-Tri-Analysis/blob/master/src/Loan%20Default%20Pipeline.py)

I decided not to oversample my imbalanced classes and instead used sample weights in AdaBoost to incentivize my cost function to focus more on the minority class. I made this decision as I wanted to use Scikit learn's Pipeline as an estimator in a GridSearchCV, in order to tune my hyper-parameters.

Although I eventually decided not to Oversample, I did however hack the pipeline functionality to exploit its book keeping while allowing me to have a re-sampling transformer step, the code can be found HERE.

### Results
Classifying the 3% signal proved to be quite a challenge. The firm told me that their cost of a False Positive, aka making a call to a customer who wasn't going to default on their loan, to a False Negative, aka not making a call to a customer who was going to default on their loan, is about 1:43.  
For this Cost-Benefit Matrix, I decided on the threshold which maximized my profit, as shown in the graph below:  

<p align="center">
  <img src='/images/determine best threshold.png' width="900"  height="550">
</p>

The optimal model had a Recall of 98%. The Precision wasn't very high which indicates that the model is overfitting for the minority class, but given the Cost-Benefit matrix this is the optimal solution for the business  


## Location Recommender:
The firm has a number of office locations spread across a state, and they're wishing to expand. These offices, although crucial to facilitating new business, are fairly expensive and hence optimizing for their location is a top priority for management. My goal was to recommend new office locations to them which would maximize growth opportunity

### Data
I got the existing office profitability data from the firm.  
I used data from India's census from 2011 to determine the district GDP

### Approach
This is a constrained optimization problem, with the objective being to find 'n' global minimas of a cost function over a Longitude-Latitude space. I defined the cost function as a linear combination of parameters as such:

<p align="center">
  <img src='/images/Cost Function.png' width="1400" height="350">
</p>

I used Scipy Optimize Basin Hopping method to traverse the cost function as it gives a lot of low level control over the optimizing function, and it does multiple random jumps to make sure we don't get stuck in a local minima. The code can be found [here](https://github.com/Namanj/Mortgage-Market-Tri-Analysis/blob/master/src/Location_Optimizer.py)

The cost function was really sensitive to initializations and one of the main challenges of defining it was scaling the data that I had collected from different sources  

### Results
The cost function strikes a balance between clustering of office locations vs spreading them out  
Here we can see the cost function is able to recommend 'n' globally optimal office locations:  
<p align="center">
  <img src='/images/UP_visualized.png'  width="700"  height="600">
</p>

The visualization code can be found [here](https://github.com/Namanj/Mortgage-Market-Tri-Analysis/blob/master/src/Location_Recommender.py)  

I add the previously accepted point into my cost function calculation after each iteration of the Basin Hopping function which ensures that the recommendations as a whole represent the 'n' best locations.  
However it's quite a challenge to determine whether these recommendations are infact globally optimal. I'm continuing to work with the firm to understand these group of recommendations in a real world context and tweak the cost function.  

## Forecasting Business:
My 3rd and final goal was to be able to predict upcoming volume of business over the next quarter so that the firm can better manage its capital reserves  

### Data
This is the amount of business the firm has done per month from 2012 to 2016:
<p align="center">
  <img src='/images/Shubham volume of business overtime.png' width="700"  height="400">
</p>

As we can see, there was a strong one time event in the middle of 2016. Such a strong one time event, this close to the horizon, is very bad from a forecasting perspective and I had to deal with it before I could do the forecasting  
### Approach
I used the ACF and PACF graphs to determine the appropriate number of AR and MA lags. As I knew that loan activity tends to have a strong yearly pattern, I decided to incorporate the seasonality explicitly by using a SARIMAX model with 12 months lag.  
I dealt with the strong one time event by replacing its values by the predictions from the best model.  
A notebook on this analysis can be found [here](https://github.com/Namanj/Mortgage-Market-Tri-Analysis/blob/master/notebooks/Time%20Series%20Forecasting.ipynb)

### Results
The SARIMAX model was pretty successful in forecasting for the 30% unseen data:
<p align="center">
  <img src='/images/Forecast.png' width="1000"  height="550">
</p>
The baseline model RMSE was 0.384 while the SARIMAX model RMSE was 0.186  

This ability to predict the upcoming amount of business will help the firm better negotiate terms for its loans, minimize excessive capital in reserves, and also minimize opportunity cost of refusing higher risk loan applicants  
## Next Steps:  
- For the Loan Default Classifier I would like to extend model to allow a particular branch to determine the risk of loan default per portfolio

- For the Location Recommender I would like to extend the Cost Function to determine the trade-off between clustering of office locations vs having a larger spread. Also, make the Cost Function less sensitive to initializations.  

- For Forecasting Business I would like to Incorporate Exponential Smoothing (ETS) in the forecasting