# Phone Pricing Tool

phone-price-tool is a Machine Learning project built to help users estimate a price for their used iPhone. 
The project leverages machine learning to learn prices from recently sold phone data and to help filter out over/underpriced sales.

Website: [https://phone-price-tool.herokuapp.com](https://phone-price-tool.herokuapp.com/)
Model Dashboard: [https://phone-price-tool.herokuapp.com/admin](https://phone-price-tool.herokuapp.com/admin)


## Model Details

This project uses Lasso Regression for scheduled training due to the inherent 
[Feature Selection](https://jdmendoza.github.io/2019/01/24/feature-selection.html) ability and parameter interpretability.
This is paired with a [Cross Validated Grid Search](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) 
to select an alpha hyperparameter which is required for Lasso Regression. 
 
 
 ## Services Used
 
 * [AWS Lambda](https://aws.amazon.com/lambda/) - Gives serverless function capabilities for model inference end-point
 * [AWS S3](https://aws.amazon.com/s3) - Cloud storage for object storage (preprocessing object, ML models) and data
 * [AWS DyamoDB](https://aws.amazon.com/dynamodb/) - NoSQL database for storing performance metrics & hyperparameters 
 * [AWS EC2](https://aws.amazon.com/ec2/) - Cloud compute to run web-scraper and schedule training scripts
 * [Heroku](www.heroku.com) - Cloud platform for web app deployment
 * [Weights & Biases](https://www.wandb.com/) - ML Experiment tracking


## Built With

* [Scikit learn](https://scikit-learn.org/stable/) - Machine Learning package 
* [Pandas](https://pandas.pydata.org/) - Used to analyze and manipulate data
* [Boto3](https://aws.amazon.com/sdk-for-python/) - Used to connect Python with AWS services (S3, DynamoDB)
* [Wandb](https://www.wandb.com/) - ML experiment tracking
* [Flask](https://flask.palletsprojects.com/en/1.1.x/) - Python web framework  
* [Dash](https://plotly.com/dash/) - Dashboard web app for model tracking
* [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) - Used for webscraping data


## Authors and acknowledgement

The project was built in 2020 by [Danny Mendoza](http://jdmendoza.github.io/).

Special thanks to [Parin Shah](https://github.com/parinpshah94) for brainstorming with me and for code reviews. 


Thanks to the open source community for providing such powerful tools.

