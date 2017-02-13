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
I decided to break this broad goal down into 3 specific goals:  
1) Build a classifier that predicts the probability that a customer will default on their loan  
2) Recommend new office locations which maximize growth potential  
3) Forecast upcoming amount of business over the next quarter  

Want a quick 4 minute overview? I gave a lightening presentation which can be found here:
<p align="center">
  <a href="https://www.youtube.com/watch?v=F0mVX1ReX2s">
  <img src='https://img.youtube.com/vi/F0mVX1ReX2s/0.jpg'>
</a>
</p>


## Loan Default Classifier:

## Next Steps:

I’m passionate about democratizing access to capital and helping people left behind the formal economy to reap the benefits of capitalism. My goal over these 2 weeks was to help the firm optimize its capital allocation, both within the firm and as loans to customers.

A significant portion of the Indian population isn’t able to participate in the formal economy for a number of reasons ranging from lack of a formal source of income, lack of filed income taxes, lack of collateral and many more. Such requirements, although well thought out, exclude too many responsible people who wish to finance an investment.

I’ve been in talks with Shubham Housing Finance ,which is located in Delhi, India, for the past couple of weeks. This firm serves the underserved portion of the population as defined previously via an in house system that rates the credit worthiness of a loan applicant and has so far given out more than USD$128 Million of loans in the past 5yrs.


I will explore 3 main high level questions:

1.	Can I make a model that predicts whether a loan receipient is going to miss their next EMI payment? This is harder said than done as our loan applicants are by definition excluded from the formal economy, and hence don’t have means to demonstrate their fiscal responsibility over time. I will hence have to rely upon less obvious metrics to measure the loan worthiness of an applicant.

2.	Predict the 3 cities where its most beneficial to open the next brick and mortar office. Currently Shubham has 90 offices in Northern India and plans on expanding. These offices, although crucial to facilitating business, are fairly expensive and hence optimizing for their location is a top priority for management. I plan to use the data both from Shubham, Indian Census from 2011 and various other web scraped sources in this analysis.

3.	Predict upcoming volume of business so that Shubham can better manage its capital reserves. If you think about it, the raw material that this firm consumes is capital itself, as this firm is in the business deploying capital based on its risk evaluation. If we can do a better job of predicting the volume of incoming business over a time period then we can negotiate better terms for our loans, minimize excessive capital in reserves, and also minimize opportunity cost of refusing higher risk loan applicants.

