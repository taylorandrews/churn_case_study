# Ride Sharing Churn Case Study

This is an industry case study. The problem is churn prediction with a
ride-sharing company in San Francisco.

I worked with 3 other data scientists to try to model and predict churn for the
ride sharing company. Our definition of a churned used that we chose was
someone who has taken a ride with the company since January (the start of the
dataset) and hasn't taken a ride during the previous 30 days.

It is a private dataset, so that has not been uploaded to the repository. I
uploaded a small subset so the code would run as-is.

## Process

There were four phases of the project.

 1. Exploratory Data Analysis

    * First, we cleaned the data. We made two datasets. One with all numerical
    values and one with some strings as data. These are labeled the numerical
    dataframe and the categorical dataframe in eda.py.

    * We partitioned our data into a set of 80% data to split into train and
    test sets. We saved 20% as a validation set for when we settle on a final
    model to test as brand new data.

    * Make simple chart of Churn vs. each feature of the data to get an idea of
    the linear relationships. These are located in the 'plots' directory in
    this repo.

    * This helped us get an idea of which features to brush through with more
    care.

 1. Preliminary Modeling

    * Curated a list of potential models - both regression models and
    categorical models.

    * Ran all eight model we settled on with the default hyperparameters from
    the scikit-learn functions we imported and compared the accuracy, precision
    and recall of the vanilla models on a test dataset.

    * Based on these initial results, we selected three of the highest
    preforming models for phase three of the case study.

 1. Model & Feature Deep Dives

    * We choose an ADABoost model, a neural net and an SVC model to look more
    closely at. We tried to improve these through manual hyperparameter grid
    searching.

    * We focused the manual grid searching on improving the recall of the
    model. The real world meaning of recall in this study is the percent of
    people that truly churn that the model correctly predicts. The ADABoost
    model turned out to be the most effective model that we could find.

    * The highest we could get the accuracy was about 79%. The precision and
    recall both ended up over 80%.

    * We looked at the effect that each feature in the dataset had on the churn
    column. Through a simple p-value analysis we found that the non-normalized
    large values naturally had a larger effect on the churn column. The
    categorical features had less significant p-values. If we were to do
    further analysis, we would normalize the values and get new p-values for
    the features.

 1. Presentation & Recommendation

    * We chose some of our most informative data visualizations and made a presentation to explain out process and findings.

    * We made some recommendations to the ride sharing company that included things like implementing coupon and reward systems focused on the users that our model predicts to churn.
