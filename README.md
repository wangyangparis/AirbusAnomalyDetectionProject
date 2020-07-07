# Airbus-Anomaly-Detection-Project.ipynb

Airbus Helicopter Anormaly Detection (Yang WANG)
Dedication:
First of all, I would like to dedicate this work to Louis Charles BREGUET, designer, builder, pioneer and inventor of modern helicopter.


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
