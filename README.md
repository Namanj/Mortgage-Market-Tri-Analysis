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
This project was developed as a capstone project for Galvanize's Data Science Immersive program.

I worked with data from Shubham Housing Finance, a firm in India that has given out more than USD $150 Million as loans over the past 5 years.  

My goal was to use data science to help the firm optimize its usage of capital, both in its loan allocation process and in its expansion.  
I decided to break this broad goal down into 3 individual more specific goals:  
1) Build a classifier that predicts the probability that a customer will default on their loan  
2) Recommend new office locations which maximize growth potential  
3) Forecast upcoming amount of business over the next quarter  

Want a quick 4 minute overview? I gave a lightening presentation which can be found here:
<p align="center">
  <a href="https://www.youtube.com/watch?v=F0mVX1ReX2s">
  <img src='/images/Presentation.png'>
</a>
</p>


## Loan Default Classifier:
First of my goals was to build a classifier that can predict the probability of a customer defaulting on their loan payment. If the firm can recognize such customers ahead of time then it can do a better job of getting ahead of the problem.

### Data
I used data provided to me by the firm which consisted of all the loans that they had given over the past 5 years, along with their performance. There were a total of ~15k data points with each representing a loan, and there were 40 columns of which 20 were categorical, 19 were numerical and 1 was temporal.

### Approach
I spent quite some time reducing the dimensionality of my data by sub-selecting the categorical variables based on signal to noise ratio, sub-selecting the numerical columns by thinking about how they would correlate with my dependent variable, and by doing some feature engineering. I eventually decided to use these features in my model:  
- a

I eventually decided to use the AdaBoost ensemble model, which I tuned using a GridSearch. The code can be found HERE

### Results
Classifying the 3% signal proved to be quite a challenge. The firm told me that their cost of making a call 

<p align="center">
  <img src='/images/determine best threshold.png' width="900"  height="550">
</p>

## Location Recommender:
The firm has a number of office locations spread across a state, and they're wishing to expand. My goal was to recommend new office locations to them which would maximize growth opportunity.

### Data
I got the existing office profitability data from the firm, and I used data from India's census from 2011 to determine the district GDP.

### Approach
This is a constrained optimization problem, with the objective being to find n global minimas of a cost function over a Longitude-Latitude space. I defined the cost function as a linear combination of parameters as such:

<p align="center">
  <img src='/images/Cost Function.png' height="450">
</p>


The cost function was really sensitive to initializations and one of the main challenges of defining it was scaling the data that I had collected from different sources.  

### Results
My cost function strikes a balance between clustering of office locations vs spreading them out.  
<p align="center">
  <img src='/images/UP_visualized.png'  width="700"  height="550">
</p>

## Forecasting Business:
Finally my 3rd goal was to be able to predict upcoming volume of business over the next quarter so that Shubham can better manage its capital reserves. If we can do a better job of predicting the volume of incoming business over a time period then we can negotiate better terms for our loans, minimize excessive capital in reserves, and also minimize opportunity cost of refusing higher risk loan applicants.

<p align="center">
  <img src='/images/Shubham volume of business overtime.png' width="700"  height="400">
</p>

<p align="center">
  <img src='/images/Forecast.png' width="900"  height="500">
</p>

## Next Steps:

I’m passionate about democratizing access to capital and helping people left behind the formal economy to reap the benefits of capitalism. My goal over these 2 weeks was to help the firm optimize its capital allocation, both within the firm and as loans to customers.

A significant portion of the Indian population isn’t able to participate in the formal economy for a number of reasons ranging from lack of a formal source of income, lack of filed income taxes, lack of collateral and many more. Such requirements, although well thought out, exclude too many responsible people who wish to finance an investment.

I’ve been in talks with Shubham Housing Finance ,which is located in Delhi, India, for the past couple of weeks. This firm serves the underserved portion of the population as defined previously via an in house system that rates the credit worthiness of a loan applicant and has so far given out more than USD$128 Million of loans in the past 5yrs.


I will explore 3 main high level questions:

1.	Can I make a model that predicts whether a loan receipient is going to miss their next EMI payment? This is harder said than done as our loan applicants are by definition excluded from the formal economy, and hence don’t have means to demonstrate their fiscal responsibility over time. I will hence have to rely upon less obvious metrics to measure the loan worthiness of an applicant.

2.	Predict the 3 cities where its most beneficial to open the next brick and mortar office. Currently Shubham has 90 offices in Northern India and plans on expanding. These offices, although crucial to facilitating business, are fairly expensive and hence optimizing for their location is a top priority for management. I plan to use the data both from Shubham, Indian Census from 2011 and various other web scraped sources in this analysis.

3.	Predict upcoming volume of business so that Shubham can better manage its capital reserves. If we can do a better job of predicting the volume of incoming business over a time period then we can negotiate better terms for our loans, minimize excessive capital in reserves, and also minimize opportunity cost of refusing higher risk loan applicants.

