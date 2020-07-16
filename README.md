# Airbus-Anomaly-Detection-Project.ipynb

<p align="center"><img src="https://github.com/savoga/data-challenge-anomaly-detection/blob/master/img/airbus_helicopters.png"></img></p>

Dedication:
First of all, I would like to dedicate this work to Louis Charles BREGUET, designer, builder, pioneer and inventor of modern helicopter.

<p align="center"><img src="https://camo.githubusercontent.com/6d3162fa2c7b7b39f75868fe82cb32510e2c476f/68747470733a2f2f7374617469632e6c706e742e66722f696d616765732f323031362f30372f30392f343635393739352d343635393739302d672d6a70675f333636303935395f31303030783636372e6a7067"></img></p>

# Data Challenge - Anomaly Detection

This is a university project in the form of a data challenge that I did during my data science degree at Télécom Paris. This readme summarizes my progress throughout the analysis. For the exact progress, see the notebook file.

## Context

The data set is provided by Airbus and consists of the measures of the accelerometer of helicopters during 1 minute at frequency 1024 Hertz, which yields time series measured at in total 60 * 1024 = 61440 equidistant time points.

## Data

### Training data

The training set consists of one file, airbus_train.csv.

File airbus_train.csv contains one observation per row, each observation having 61440 entries, measures with equivalent time distance of 1 / 1024 seconds.

There are in total 1677 training observations.

### Test data

The training set consists of one file, airbus_test.csv, which has the same structure as file airbus_train.csv.

There are in total 2511 test observations.

## Introduction

For this challenge, I've chosen first to perform some basic analysis to understand the data. This analysis consists of displaying train/test data in order to spot the differences. This allowed me to draw conclusions on the difference of both datasets e.g. the presence of outliers, standardization, stationarity,... and to have a first idea of interesting features to use for the future of my work.

Then, I tried a large number of different algorithms. It turns out that simple Feature Engineering worked the best. I thus focused on this method using the best features as possible. Important to note that the package *tsfresh* brought significant value to my research.

## 1. Data exploration

#### Autocorrelation

5 first entries (columns):

<p align="center"><img src="https://github.com/savoga/data-challenge-anomaly-detection/blob/master/img/first_entries.png"></img></p>

5 random entries:

<p align="center"><img src="https://github.com/savoga/data-challenge-anomaly-detection/blob/master/img/random_entries.png"></img></p>

Since random entries give much more volatile trends, our first conclusion is that autocorrelation may be a good feature to consider (which is often the case with time series).

#### Standardization

Looking at standardization is important to make sure all features contribute equally to the analysis.

After sampling few data on both sets, we can plot few distributions:

<p align="center"><img src="https://github.com/savoga/data-challenge-anomaly-detection/blob/master/img/standardization_all.png"></img></p>

Graphs from train set (left) indicates that data are standardized.
We can draw the same conclusion for the graph of the test set, however a possible presence of outliers give more spread values (higher variance). This will be confirmed in the next section.

#### Outlier presence

The PCA (see explanation below) allows us to display observations in 2 dimensions. We can thus easily spot outliers on the test set (right).

<p align="center"><img src="https://github.com/savoga/data-challenge-anomaly-detection/blob/master/img/pca_all.png"></img></p>

#### Stationarity

2 random observations from the train set:

<p align="center"><img src="https://github.com/savoga/data-challenge-anomaly-detection/blob/master/img/stationarity_train.png"></img></p>

We can see that the both series seem quite stationary. It's less obvious for random observations for the test set:

<p align="center"><img src="https://github.com/savoga/data-challenge-anomaly-detection/blob/master/img/stationarity_test.png"></img></p>

Although the first observation can seem stationary, the mean is quite high compared to most of the observations (as seen previously) which makes it an outlier.

## 2. Data augmentation

Since both datasets have a relatively low amount of observations, it can be useful to consider data augmentation methods to grow our datasets. There are numerous ways of doing so with time series, I tried few of them.

#### Adding random noise

The easiest way to perform data augmentation is to add a noise to the data. Concretely, it is done as such:

```
mu, sigma = 0, 0.1
noise = np.random.normal(mu, sigma, [data.shape[0],data.shape[1]])
data_noise = data + noise
data_extend = np.concatenate((data, data_noise), axis=0)
```

Although this is fairly simple, it allowed me to slightly boost my detection score.

#### Other methods

Additional methods include drifting, random time warping, pooling,... I've used briefly an interesting package called <a href="https://pypi.org/project/tsaug/">tsaug</a>.

## 3. Dimension reduction

As the datasets are large, some detection algorithms would require to reduce the dataset first. This can be done using adapted dimension reduction methods.

#### PCA

The <a href="https://en.wikipedia.org/wiki/Principal_component_analysis">PCA</a> allows us to reduce the dimensions of both datasets into components that best explain the variance. The graph below shows that the first 20 components seem to explain most of the variance.

<p align="center"><img src="https://github.com/savoga/data-challenge-anomaly-detection/blob/master/img/pca_variance.png"></img></p>

We can thus perform our analysis on the first 20 components given by the PCA.

#### Autoencoder

<a href="https://en.wikipedia.org/wiki/Autoencoder">Autoencoder</a> is a more recent algorithm that can be used to perform dimension reduction. It is based on neural networks and can find complex separation functions (whereas PCA is for linear separation only).

<p align="center"><img src="https://github.com/savoga/data-challenge-anomaly-detection/blob/master/img/autoencoder.png" width=500></img></p>

In the case of a dimension reduction, only the bottleneck (latent space) is relevant for us.

Surprisingly, PCA gave better results when combined with detection algorithms.

## 4. Detection algorithms

In order to detect outliers, plenty of algorithms are already implemented and quite easy to use.

#### Stationarity test

A famous statistic test for testing stationarity is the <a href="https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test">Augmented Dickey-Fuller test</a>. Essentially, it tests the presence of a unit root.

My idea was to compute the test on each observation and score it based on the *p-value*. Doing so on few observations gave quite promising results.

<p align="center"><img src="https://github.com/savoga/data-challenge-anomaly-detection/blob/master/img/adf_ts.png"></img></div>

```
ADF p-value: 0.0
ADF p-value: 1.4514797225057198e-23
```

The p-value is higher of the second serie so there is a higher chance to accept the unit root hypothesis; the serie is more likely to be not stationary.

The problem is that computing the test on all the observations was quite a pain; the series are too long and it makes the test computationally not feasible. Alternatively, I tried to compute the test on reduced data (after PCA or smoothing). But the results were not so good.

#### Autoencoder

Autoencoders can be used to learn about a distribution and reconstruct some data. The method consists of learning on the train set and scoring on the test set using the following loss function:

<p align="center"><img src="https://github.com/savoga/data-challenge-anomaly-detection/blob/master/img/loss-autoencoder.gif"></img></p>

Observations associated with a higher loss will be more likely to be outliers since they are likely to be very different from the train set.

#### Isolation forests

<a href="https://en.wikipedia.org/wiki/Isolation_forest">Isolation forests</a> are quite intuitive to detect outliers. In short, they allow to isolate abnormal data that have significantly different attributes.

This method gave first satisfactory results when combined with PCA with 20 components.

#### Local Outlier Factor

<a href="https://en.wikipedia.org/wiki/Local_outlier_factor">Local Outlier Factor</a> is a quite simple detection algorithm that performed really well for this problem. It aims at comparing local densities of test observations with train observations. 

Local densities are based on reachability-distance defined as such:

<p align="center"><img src="https://github.com/savoga/data-challenge-anomaly-detection/blob/master/img/reachability-distance.gif"></img></p>

The <img src="https://github.com/savoga/data-challenge-anomaly-detection/blob/master/img/k-distance.gif"></img> is the distance between an object and its kth nearest neighbor.

The reachability-distance tells us how far is a point starting *from* its neighbors.

The distance metric I used is simply the *euclidean* distance. One of the main advantage of using such an easy metric is that I could easily introduce binary variables when doing feature engineering.

#### Score averaging

I tried to run several models and then averaging their scores but the results were not satisfactory.

## 5. Feature engineering

Feature engineering played a huge part in this project as I could achieve significantly better results using relevant feature.

During this challenge I found out a library called <a href="https://tsfresh.readthedocs.io/en/latest/">tsfresh</a> that I found very powerful. It has more than 60 features implemented for time series.

The 5 features that gave me the best scores are the following:

- autocorrelation (lag 3)

- mean

- max

- c3: <img src="https://github.com/savoga/data-challenge-anomaly-detection/blob/master/img/c3.gif"></img>

- cid: <img src="https://github.com/savoga/data-challenge-anomaly-detection/blob/master/img/cid.gif"></img>
  
- symmetry looking: <img src="https://github.com/savoga/data-challenge-anomaly-detection/blob/master/img/symmetry-looking.gif"></img>
  
The last three features allow to measure linearity, complexity and symmetry of a serie. More details can be found on the library website.

## 6. Approaches

I used various approaches for this project:

Table of Contents

0 Airbus Helicopter Anormaly Detection  
0.1 Challenge Large Scale Machine Learning
0.2 Functional anomaly detection
0.3 The properties of the dataset:
0.4 The performance criterion: 0.5 Import
0.5.1 Load and investigate the data
0.5.2 A sample plot 1 Some physics before start

1.1 What can be the anormalies?
1.1.1 Aircrafte Flutter
1.2 What are the possible approaches?
1.2.1 Raw signal time series observation 1.2.2 Time Series Clustering
1.2.3 Similarity
1.2.4 DTW (Dynamic Time Warping)
1.3 Periodogram-based distance
1.4 First order/seconde order derivative and other feature engenieering 1.5 Build fonctionnal space
1.6 Interpolation - Spline
1.7 1st , 2nd order derivative

2 Frequency Domaine 
2.1 Perodogram
2.2 FFT Fast Fourier Transformation
2.3 STFT Short-term Fourier Transformation 2.4 Time–frequency analysis

3 Frequency Domaine approach with STFT Matrix based 
3.1 2 D and 3 D STFT Matrix
3.2 STFT with Univariate Time Series treatment
3.2.1 LOF - on STFT 61-dimensional space
3.2.2 LOF on a STFT PCA 10-dimensional space 3.3 3D STFT Matrix and LSTM - Multivariate Time Series 3.4 3D STFT VAE conv2D
3.4.1 Train VAE on Frequency Domain 3.4.2 Reconstruction Error
3.4.3 VAE Latent Space Visualisation
3.4.3.1 IsolationForest 

4 Raw Signal Time Series Approach
4.1 LSTM
4.2 LOF + PCA
4.3 Kernel PCA
4.4 OneClassSVM
4.5 Isolation Forest Raw data
4.6 LOF + PCA + Derivatives
4.7 LOF interpolated data
4.8 Isolation Forest Interpolation / 1st order derivative 4.9 Autoencoder with data +der1+der2
4.10 VAE data+der1+der2
                                                                                                                                                  
4.11 LOF - 1st order derivative and 2nd order derivative 5 Data Augementation

5.1 Standartized data
5.2 construction of new features

6 Statistiques Feature Engineering Approche
6.1 Baseline models with hyperparameter tuning
6.2 New Features Construction
6.3 Combine raw data + new features
6.4 Feature construction with 1st/2nd order derivatives and FFT 6.5 Test new features

7 History
7.1 Prepare a file for submission
7.2 Best score:79% ensemble OCSVM+Isolation Forest
