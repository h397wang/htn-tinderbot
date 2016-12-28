# htn-tinderbot

By: Phil, Richard and Henry

Hack the North 2016

Tinder Bot Script

Design

Uses the pynder API to programatically perform actions on the Tinder Application.
The python script fetches the profiles of nearby users and selects appropriate matches based on certain criteria chosen by the user.
Using OpenCV's library, the region of interest containing the face is passed as a parameter to our script running an artificial neural network.
The neural network implementation was developed with help from the Lasagne python library.
Our training set consisted of a total of 3000 images of female facial profiles with different basic classifications.
All images within the data set were fetched from Google Images using a basic (modified) python script.
Upon matching with another user, a random pick up line fetched from a text file is sent to the user.
Have the script running on a web server such that it would be accessible by anybody.
Users of the web application would be able to select their preferences and have the automation do the job.

Results

The neural network had high bias. With test sets it "swipes right" on every profile. Although we were able to achieving around 80% success rate during training of the neural network with our training set, the function did not perform as well with actual Tinder profiles. With the short timeframe of the project we were not able to develop a more complex model but an overly simplistic model is likely the issue here.

Hindsight

A neural network is more appropriate for detecting whether or not an object is present, not for distinguishing subtle details.
Consider other features of the face, such as the ratio of various facial elements.
Learn how to do web dev so we can run this program somewhere other than our machine.
