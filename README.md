# Football Analytics

The aim of the following report is to describe the study carried out on the application
of Machine Learning, namely Neural Networks, for the prediction of football match results
based on the past performance of the involved teams.

Large amounts of data have been collected during matches by TV channels, football clubs
and betting agencies in the last few years. This enables the detailed analysis of the players
and teams performance, and therefore the creation of predictive models of their future
performance, and consequently the results of the matches. Nowadays there are still few
research studies on the matter, but increasingly more researchers are deciding to apply
their knowledge in the domain of sport analytics.

This document describes the complete process that has been followed, as well as the different
techniques that have been used to create a new database, process the data, perform
feature engineering, build a basic baseline model trained with the bookmaker odds, and
finally build the final Bagging Neural Networks based model. All this after a revision
and critique of the previous relevant related literature.

In order to develop the model, first of all, Multi-layer Perceptron basic models have been
trained by seeking the optimal hyper-parameters, as well as searching for the features
that better train our model. From the obtained results, more complex models have been
trained, to finally obtain a model based on Bagging Neural Networks, where we have X
independent neural networks trained with different hyper-parameters and with different
input data. At the end we average their predictions and get a final prediction of whether
the home team wins, the away team wins or the result is a draw.
