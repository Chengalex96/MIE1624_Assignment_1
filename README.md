# MIE1624_Assignment_1
 
I am using Kaggle Dataset to examine the nature of women's representation in Data Science and Machine Learning. 
Also to see the effects of education on income level

**Getting Started**
- Install all of the required libraries
- Convert dataset to pandas dataframe, the dataset is composed of 10729 rows and 356 columns (35 questions with multiple parts 1-11)
- The df.describe() function shows an overall summary such as the count, mean, std, min, max, and percentile

**Question 1: Exploratory Data Analysis to Analyze the Dataset**
- Can analyze where most survey participants are from: country.value_counts()[:10].index.tolist()
- Can find out the age, education levels, professions, salaries

![image](https://github.com/Chengalex96/MIE1624_Assignment_1/assets/81919159/e678b440-c43d-478c-9abc-34bcb1ec744f)
![image](https://github.com/Chengalex96/MIE1624_Assignment_1/assets/81919159/f76a4fd2-cce3-40ab-bf57-95aa0614ded9)
![image](https://github.com/Chengalex96/MIE1624_Assignment_1/assets/81919159/bf819595-e17e-4996-8693-228b8d033ecb)

- Plot two variables to see the trend: plt.plot(df.groupby(['Q1'])['Q24'].mean())

![image](https://github.com/Chengalex96/MIE1624_Assignment_1/assets/81919159/62885ea6-1004-440c-9d5c-9f7750f75d5b)
![image](https://github.com/Chengalex96/MIE1624_Assignment_1/assets/81919159/a65c8869-765c-4f02-a42a-12e3be43d6bb)
[image](https://github.com/Chengalex96/MIE1624_Assignment_1/assets/81919159/a21c902a-9c84-4636-8286-91dcf51b20b8)

**Question 2: Explore the difference between the average salary of men vs women **

Create a new dataframe with only gender and salary, then groupby gender: adf.groupby('Q2')

![image](https://github.com/Chengalex96/MIE1624_Assignment_1/assets/81919159/83ead765-63c4-42d1-b988-689dbc69dab1)

Isolate the male and female salaries: female_salary = adf[adf['Q2'] == 'Woman']['Q24']

![image](https://github.com/Chengalex96/MIE1624_Assignment_1/assets/81919159/77d6f59c-5ac1-4c67-94a8-22f471496f66)

The distribution does not fit a normal distribution so we cannot run a 2 sample t-test. Will perform a 2-sample t-test even though it doesn't follow the assumptions required:
Assumptions: 
- 2 groups are independent, normally distributed, and have similar variance
- Our data has similar variance and can assume independence, however, data is not normally distributed, We cannot run t-test
- The null hypothesis states that the difference in group means is zero

tc, pc = stats.ttest_ind(female_salary, male_salary)

print ("t-test: t = %g  p = %g" % (tc, pc))

Since p = 4.77e-15 << 0.05, it is statistically significant), we reject the null hypothesis. We can observe a clear relationship between gender and relationship. Women are paid less than their male counterparts. This is the case if data was normally distributed, since it isn't, the t-test may not be valid since we're using the wrong assumptions

Bootstrapping the data for the data to converge to the central limit theorem and fitting a normal distribution.

Procedure: Randomly select a sample, repeating that multiple times (8872 for males salary) to calculate the mean, and then plotting the distribution of those means. Bootstrap means to take a random sample with replacement, ie take a copy, not removed from the list.

![image](https://github.com/Chengalex96/MIE1624_Assignment_1/assets/81919159/a80d9072-14ae-42d6-a8a0-9da57f4bd9cc)
![image](https://github.com/Chengalex96/MIE1624_Assignment_1/assets/81919159/5d4b962c-032d-433e-8ab5-17ab95a5b40b)
![image](https://github.com/Chengalex96/MIE1624_Assignment_1/assets/81919159/eefee88d-6afa-4016-93f9-71876e07ad0d)

Calculate the difference in salaries and plot the histogram of the bootstrapped data.

**Summarize Findings: **

After performing the t-test: After applying bootstrapping to the male and female salaries, the mean salary data was normally distributed. This allowed us to perform a 2-sample t-test. Using the 2 sample t-test, we got a p-value equal to 0, which shows the probability of the men's salary mean and the women's salary mean being equivalent. Thus, we can conclude there is a relationship between gender and salary. Women are being underpaid compared to their male counterparts if we solely look at gender and salaries. 

**Question 3: Analyzing the relationship between education and salary**

Create a new dataframe with only education and salary, then groupby education: adf.groupby('Q4')

![image](https://github.com/Chengalex96/MIE1624_Assignment_1/assets/81919159/52f7cc6c-f664-4380-a7c6-a1fd187d0d8d)

ANOVA testing - check if it's suitable to use by checking the distribution first
- Scipy.stats.shapiro( ) can test the null hypothesis that the data was drawn from a normal distribution.
- It returns a tuple of test statistics and p-values. If the p-value is less (<) than the alpha(0.05), we reject the null hypothesis, which means the data is not normally distributed.

Since the dataset is not accorded to a normal distribution, ANOVA test can not be conducted. We perform the same bootstrapping methods for each education level: Bachelor, doctoral, and master's.

![image](https://github.com/Chengalex96/MIE1624_Assignment_1/assets/81919159/d8fc9a93-bbec-4f2e-9784-8a6f5dd6214d)

Perform 1-way ANOVA after bootstrapping
- Since p = 0 < 0.05, it is statistically significant, we reject the null hypothesis
- We can observe a clear relationship between education and salaries. Those who've completed a doctoral degree gets paid more, then masters, then bachelor's

tc, pc = stats.f_oneway(bachelor, doctoral, master)

print ("f-test: f = %g  p = %g" % (tc, pc))

Comparison of histogram in 'Salaries' and the histogram of difference
- The orange line is the difference in salary between doctoral and masters
- The green line is the difference in salary between doctoral and bachelor's
- The red line is the difference in salary between a master's and bachelor's
  
![image](https://github.com/Chengalex96/MIE1624_Assignment_1/assets/81919159/5a92108b-f930-4ecd-8ae3-debeea314e53)

**Summarize Findings:** 
After applying bootstrapping to the doctoral, master's, and bachelor's salaries, the mean salary data was normally distributed. This allowed us to perform a 1 way ANOVA test. Using the 1 way ANOVA test, we got a p-value equal to 0, which shows the probability of the doctoral salary mean, the master's salary mean, and the bachelor's salary mean being equivalent. Thus, we can conclude there is a relationship between education level and salary. Those who obtained their doctoral degree are being paid higher compared to if they have received a master's or bachelor's degree. This is if we solely look at education and salaries, not accounting for country, job position, etc.
