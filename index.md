### 2024

### [PDF Documents Manager](https://github.com/DanOKeefe/documents-manager)
  - Web application for managing documents.
  - Generates summaries for documents after upload.
  - Documents stored in S3, metadata and summaries stored in DynamoDB.
  - Pages for viewing existing documents and uploading new documents.

### 2023

### [Effective Data Handling with Custom PyTorch Dataset Classes](https://dantokeefe.medium.com/effective-data-handling-with-custom-pytorch-dataset-classes-b141bcb87b41)
  - Writing custom PyTorch [Dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) classes to handle data during training.
  - Separating data processing code from training code, improving readability.
  - Enabling training on datasets that don't fit into memory.
  - Utilizing transfer learning with a [vision transformer model](https://arxiv.org/pdf/2010.11929.pdf) (vit_b_16).
  - GitHub repo containing two examples can be viewed [here](https://github.com/DanOKeefe/pytorch-custom-datasets).
    - Titanic dataset example
    - Food-101 dataset example

### 2021

#### [Sentiment Classification with BERT](https://github.com/DanOKeefe/BERT_Sentiment_Classification/tree/main)
  - Create a sentiment classifier trained on a dataset of restaurant reviews from [Yelp](https://raw.githubusercontent.com/mayank100sharma/Sentiment-Analysis-on-Yelp-Reviews/master/yelp.csv).
  - Use a compact, pre-trained BERT model presented by Google Research in [this paper](https://arxiv.org/pdf/1908.08962.pdf) as a base. [This model](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4) is much smaller than the original BERT model, allowing it to easily fit in the 1 GB RAM limit of [Streamlit Sharing](https://streamlit.io/sharing), which I use to host the model in a web app.
  - Tokenize the reviews, send them through the BERT model, and retrieve the output vectors in the [CLS] position.
  - Train a classifer to take these output vectors and classify them as positive or negative.
  - Create a web app that serves the model.

#### [Deploy a Machine Learning Model with FastAPI](https://dantokeefe.medium.com/deploy-a-machine-learning-model-with-fastapi-7a98bf7cb7c3)
  - Create a RESTful API that serves a machine learning model.
  - Version 1 allows users to pass inputs as path variables.
  - Version 2 allows users to pass inputs as a JSON payload.
  - [FastAPI](https://fastapi.tiangolo.com/) automatically generates interactive documentation for the API using [Swagger UI](https://swagger.io/tools/swagger-ui/) and [Redocly](https://redoc.ly/).
  - GitHub repo for this project can be viewed [here](https://github.com/DanOKeefe/Iris_FastAPI).

#### [Image Classifier with MobileNet](https://share.streamlit.io/danokeefe/image-classifier/main/image_app.py)
  - Multi-page web application that allows users to classify an image of an object. 
  - Provides two methods of image input. The user can upload an image or provide a URL to an image online.
  - The web app processes the image for input into [MobileNet](https://arxiv.org/abs/1704.04861) and captions the image with the output prediction.
  - GitHub repo for this project can be viewed [here](https://github.com/DanOKeefe/Image-Classifier/tree/main).
  - Model is trained on [1,000 classes](https://github.com/DanOKeefe/Image-Classifier/blob/main/imagenet_classes.txt).

#### [Create a Machine Learnining App with Streamlit](https://share.streamlit.io/danokeefe/streamlit_iris_classifier/main/iris_app.py)
  - Web application developed with Streamlit to classify Iris flowers.
  - User can customize model input values using sliders in the sidebar.
  - Visualizes confidence of the model's prediction with [Plotly](https://plotly.com/python/plotly-express/).
  - Visualizes the user input compared to other datapoints on a [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) plot.
  - GitHub repo for this project can be viewed [here](https://github.com/DanOKeefe/Streamlit_Iris_Classifier).
  - Article showing how I built this application [here](https://medium.com/geekculture/create-a-machine-learning-web-app-with-streamlit-f28c75f9f40f).
  - Article was included in Streamlit's [Weekly Roundup](https://discuss.streamlit.io/t/weekly-roundup-theming-tutorials-astronomy-pictures-combining-multiple-spreadsheets-and-more/11466)

### 2020

#### [Near Real-Time Financial News Dataset with AWS Lambda](https://dantokeefe.medium.com/creating-a-near-real-time-financial-news-dataset-with-aws-lambda-509e2fe53261)
- Create an automated serverless function with [AWS Lambda](https://aws.amazon.com/lambda/) to scrape and parse [CNBC Finance](https://www.cnbc.com/finance/) articles daily.
- Use a deployment package with Python libraries not included in the AWS Lambda Python 3.8 environment.
- Save text files in an [S3](https://aws.amazon.com/s3/) bucket organized by year, month, and day, making it easily queryable from [AWS Athena](https://aws.amazon.com/athena/).

#### [Iris Flower Classifier Web Application](https://danokeefe.github.io/iris.html)
- Web application developed with [TensorFlow.js](https://www.tensorflow.org/js) that trains a feed-forward neural network within client's browser.
- Enables the client to run live inference on custom inputs.
- Visualizes confidence of the model's predictions using [Plotly.js](https://plotly.com/javascript/).
- GitHub repo for the project can be viewed [here](https://github.com/DanOKeefe/Iris-Flower-Classifier-TF.jshttps://danokeefe.github.io/iris.html).

#### [COVID-19 Dashboard](https://github.com/DanOKeefe/COVID-19-Dashboard)
- Web application for visualizing COVID-19 case density across the U.S. and Florida
- Developed with Python webframework [Flask](https://flask.palletsprojects.com/en/1.1.x/).
- Interactive visualizations created with Python library [Bokeh](https://bokeh.org/).
- Retrieves data from Johns Hopkins repository.
- Ran on a VM with [Google Cloud Platform](https://cloud.google.com/).

#### [Deploying a Portfolio Investment Strategy in the Cloud](https://dantokeefe.medium.com/deploying-a-portfolio-investment-strategy-in-the-cloud-415ef70ffdfb)
- Deploy a [Global Minimum Variance portfolio](https://faculty.washington.edu/ezivot/econ424/portfolioTheoryMatrix-BEAMER.pdf) investment strategy with daily rebalances on [Alpaca brokerage](https://alpaca.markets/) with a serverless function on AWS.

#### [Disease Classification from Blood Samples](https://danokeefe.github.io/HCV.html)
- Predict if a patient has Hepatitis, Fibrosis, Cirrhosis, or no disease given lab readings from their blood sample.
- Uses an ensemble of models: dense neural network, gradient boosted decision tree, and random forest.
- [Keras](https://keras.io/), [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_intro.html), and [scikit-learn](https://scikit-learn.org/).

#### [Plotting stock prices in Python](https://dantokeefe.medium.com/plotting-stock-prices-in-python-2b2aafaac5eb)
- Plotting OHLC data using candlesticks

### 2019

### 2018

#### [Tital Survival Prediction](https://github.com/DanOKeefe/TitanicPrediction/blob/master/Titanic_Prediction_v2.ipynb)
- Predict if a Titanic passenger will survive.
- Data cleaning and preparation.
- Feature engineering.
- Use a neural network developed with TensorFlow.

#### [MNIST Digit Classification in R](https://github.com/DanOKeefe/KerasMNIST/blob/master/cnnMNIST.R)
- Classify an image of a hand-written digit with classes 0-9.
- Utilize [Keras](https://keras.rstudio.com/) in R.

### 2017

#### [Breast Tumor Classification](https://github.com/DanOKeefe/BreastTumorClassification/blob/master/BreastCancerNormalizeInputs.ipynb)
#### [Bank Direct Marketing Campaign](https://github.com/DanOKeefe/BankDirectMarketingCampaign/blob/master/BankMarketing.ipynb)

### Notes

#### [Machine Learning Notes](https://danokeefe.github.io/ml_notes.html)
- Explanations and demos of various machine learning topics and techniques.

#### [Finance Notes](https://danokeefe.github.io/finance.html)
- Retrieving and streaming data with financial APIs and websockets
- Portfolio construction and risk analysis
- Options pricing analysis
- Backtesting investment strategies
- Implementing investment strategies with Alpaca
- Streaming live trading data to AWS

### Certifications
- [Introduction to Deep Learning](https://coursera.org/share/bc6828c2a0b3a78b01c0644fb70bdb58)
- [Neural Networks and Deep Learning](https://coursera.org/share/0525e529ea1c810a9892b2567b0a82b4)
- [Structuring Machine Learning Projects](https://coursera.org/share/508b89ec192f089d4e3bac37bbbb690c)
- [Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization](https://coursera.org/share/81f67f65646f32bfe05e171f25f90f3d)
- [Convolutional Neural Networks](https://coursera.org/share/1852e7fca212c7f0cd456ba1a9d0292b)
- [Bayesian Statistics: From Concept to Data Analysis](https://coursera.org/share/367a2548080bb46179558128d5b53892)
- [Bayesian Methods for Machine Learning](https://coursera.org/share/7cfa6f82e62737ef0dbc58898287a5cc)
- [Data Visualization](https://coursera.org/share/71f11f75306fff0faa5eb6fb05a78b0f)
- [Google Cloud Platform Fundamentals: Core Infrastructure](https://coursera.org/share/3d948d6ade21d6b0dbbf4b0c07fc0171)
- [How to Win a Data Science Competition: Learn from Top Kagglers](https://coursera.org/share/1fc1342d60e64d2c6dca87756e78d639)
- [Introduction to Portfolio Construction and Analysis with Python](https://coursera.org/share/068a2f64c107a37cc575dfdd66ee645b)
- [Advanced Portfolio Construction and Analysis with Python](https://coursera.org/share/f5c2fb4cbb6c22b41fbd6df51ba6dddb)
