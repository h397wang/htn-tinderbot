# htn-tinderbot

By: Phil, Richard and Henry

Hack the North 2016

Tinder Bot Script

#Design

Uses the pynder API to programatically perform actions on the Tinder Application.
The python script fetches the profiles of nearby users and selects appropriate matches based on certain criteria chosen by the user.
In this case the critera input is set of profile images containing faces. Using OpenCV's library, the region of interest containing the
face is passed as a input to the neural network (NN).

The neural network implementation was developed with help from the Lasagne python library.
To put it simply, this model required the a "training set". with the "answers" provided. The NN would then extrapolate or interpolate
other inputs. 

Our training set consisted of a total of 3000 images of female facial profiles that either met the classification or did not meet the
classification. All images within the data set were fetched from Google Images using a basic (modified) python script. 

The input to the NN is just the set of pixels that compose the image. All images are greyscaled, and the area containing the face is 
cropped to a predefined constant size. 

The NN output can be two values, 1 for a classification match, or 0 for no classification match. Based on the NN output, Tinder users
are selected programatically using the Pynder API.

Upon matching with another user, a random "pick up" line fetched from a predefined text file is sent to the user.

#Results

Our program "swiped right" on every user profile. Although we were able to achieving around 80% success rate during "training" of the
neural network, the function did not perform as well with actual Tinder profiles.

#Hindsight

In theory, our method of using the NN really should not have worked. Our NN basically looked at the set of individual pixels as inputs. 
The problem is that in our context of computer vision this doesn't make any sense. To analyse images, we really should be looking at 
clusters, or blocks of pixels and interpreting them together, not examining each pixel as some independent input. 

A neural network is more appropriate for detecting whether or not an object is present, not for distinguishing subtle details. All our 
images had faces, but true classification would have to examine the other features of the face, such as the ratio of various facial
elements.

Learn how to do web dev so we can run this program somewhere other than our machine.
Have the script running on a web server such that it would be accessible by anybody.
Users of the web application would be able to select their preferences and have the automation do the job.

