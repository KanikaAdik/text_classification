Objective - 
To build a replicaof classifier set for song sentiment prediction. 

https://developers.google.com/machine-learning/guides/text-classification/

https://www.sciencedirect.com/topics/computer-science/text-classification


Step 1: Gather Data
Step 2: Explore Your Data
Step 2.5: Choose a Model*
Step 3: Prepare Your Data
Step 4: Build, Train, and Evaluate Your Model
Step 5: Tune Hyperparameters
Step 6: Deploy Your Model


text classification is to classify documents into predetermined categories, typically applying machine learning algorithms. Generally speaking, organizing and using the huge amounts of information, which exist in unstructured text format, is one of the most important techniques. Classification of text is a widely studied field of language processing and text mining study. A document is represented in traditional text classification as a bag of words in which the words terms are cut from their finer context, that is, their location in a sentence or in a document. Only the wider document context is utilized in the vector space with some type of term frequency information. 



1. zero_shot_class.py -
This code uses an existing Transformer API.
Many NLP tasks have a pre-trained pipeline ready to go using Transformer.Since it is a pretrasined model we are using only for testing purposes and how it has been or can be used for sentiment analysis in any applcaition like Spotify or Musixmatch lyrics website. The Transformer is built on JAX, PyTorch and Tensorflow 
We have given it a set of Classifiers which will be used to categorise the tokens.
Finally the Transformer will categorize the data from song lyrics and determine the song sentiments based on the labels percentage for every song
This is automated already hence no tokeniser or classifier is used. 
Transformers are built to classify the datasets according to the pretrained words from various sources.
The same words when added as the categories/ labels the song lyrics sent as test data is categorised according to the labels. 


This code can be run only using Google Collab since it requires high GPU processign power

