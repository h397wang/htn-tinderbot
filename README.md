# htn-tinderbot

By: Phil, Richard and Henry

Hack the North 2016

Tinder Bot Script

Design

Uses the pynder API to programatically perform actions on the Tinder Application. 
The python script fetches the profiles of nearby users and selects appropriate matches based on certain criteria.
The criteria chosen to be evaluated was the ethnic group of the user.
In particular the script was looking for users of East Oriental descent.
Using OpenCV's library, the region of interest containing the face is passed as a parameter to a neural network function. 
The neural network implementation was attempted using the Lasagne python library.
Our training set consisted of a total of 3000 images of female facial profiles.
All images within the data set were fetched from Google Images using a basic (modifided) python script. 
Upon matching with another user, a random pick up line fetched from a text file is sent to the user. 
Have the script running on a web server such that it would be accessible by anybody.
Users of the web application would be able to select their preferences and have the automation do the job.

Results

Current function is literally just a Tinder bot.
The neural network function "swipes right" on every profile. 
Although achieving a high success rate during training/testing of the neural network, it was just good at distinguishing our positive and negative data set.

Hindsight 

A neural network is more appropriate for detecting whether or not an object is present, not for distinguishing subtle details.
The process of identifying the ethnic group of an individual through computer vision is definitely a much more complex task than a neural network function can perform. 
Consider other features of the face, such as the ratio of various facial elements.
Learn how to do web dev so we can run this program somewhere other than our machine. 


