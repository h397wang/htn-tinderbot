import random
import os
import time
import sys
import pynder
import requests
import numpy as np
import cv2
from PIL import Image
from PIL import ImageOps
from PIL import ImageFilter
from io import BytesIO
import pickle
import theano
import theano.tensor as T
import lasagne

# get the token from here
# https://www.facebook.com/dialog/oauth?client_id=464891386855067&redirect_uri=fbconnect%3A%2F%2Fsuccess&scope=basic_info%2Cemail%2Cpublic_profile%2Cuser_about_me%2Cuser_activities%2Cuser_birthday%2Cuser_education_history%2Cuser_friends%2Cuser_interests%2Cuser_likes%2Cuser_location%2Cuser_photos%2Cuser_relationship_details&response_type=token&__mref=message_bubble
# then submit it over here 
# https://developers.facebook.com/tools/explorer/?method=GET&path=me%3Ffields%3Did%2Cname&version=v2.7

AUTO = False # allows the script to automate it, otherwise requires user input

token = "EAAGm0PX4ZCpsBAGTssnzEYAM79euou3aWAPSzm3fdppEBaS72yN67NVjwni17pccg7OL9RCEiUtCgN1q4wrWLRCTdjkPL9j2un5rehAUdhwdvDlLeNtJP58YZCtd6YxHO0PEXLuzXRZAFmttmHsdoXwhxZArHOYQVxIjAFmEzwZDZD"
id = "591469779"

path = "/home/henry/opencv/data/haarcascades/"

IMAGE_SIZE = 320 # size of the img from fetched from Tinder's url request

PIXELS = 28 # pixel size that the NN uses

# detectMultiScale parameters
SCALE_FACTOR = 1.1
MIN_NEIGHBOURS = 6
MIN_SIZE = 30

# printing text parameters
BOT_LEFT = (int(IMAGE_SIZE*0.7),int(IMAGE_SIZE*0.3))
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SIZE = 5
COLOUR = (0,0,0)
LINE_THICKNESS = 15

THRESHOLD = 0.5 # how asian are they?

def build_mlp(input_var=None):
	print("Building network ...")
	l_in = lasagne.layers.InputLayer(shape=(None, 1, PIXELS, PIXELS), input_var=input_var)
    # l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)
    # l_hid = lasagne.layers.DenseLayer(
        # l_in, num_units=800,
        # nonlinearity=lasagne.nonlinearities.sigmoid,
        # W=lasagne.init.Normal())

    # Apply 20% dropout to the input data:
	l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
	l_hid1 = lasagne.layers.DenseLayer(
		l_in_drop, num_units=800,
		nonlinearity=lasagne.nonlinearities.rectify,
		 W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
	l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    # Another 800-unit layer:
	l_hid2 = lasagne.layers.DenseLayer(
		l_hid1_drop, num_units=800,
		 nonlinearity=lasagne.nonlinearities.rectify)

    # We'll now add dropout of 50%:
	l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.3)

    # Another 800-unit layer:
	l_hid3 = lasagne.layers.DenseLayer(
		l_hid2_drop, num_units=100,
            nonlinearity=lasagne.nonlinearities.sigmoid)

    # 50% dropout again:
	l_hid3_drop = lasagne.layers.DropoutLayer(l_hid3, p=0.5)

	l_out = lasagne.layers.DenseLayer(
		l_hid3_drop, num_units=2,
		nonlinearity=lasagne.nonlinearities.softmax)

    # l_in = lasagne.layers.InputLayer(shape=(None, 1, PIXELS, PIXELS), input_var=input_var)
    # l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)
    # l_hid = lasagne.layers.DenseLayer(
        # l_in_drop, num_units=800,
        # nonlinearity=lasagne.nonlinearities.rectify,
        # W=lasagne.init.GlorotUniform())
    # l_out = lasagne.layers.DenseLayer(
            # l_hid, num_units=2,
            # nonlinearity=lasagne.nonlinearities.softmax)

	return l_out

def main():
	
	input_var = T.tensor4('inputs')
	target_var = T.ivector('targets')	
	network = build_mlp(input_var)
	# load the training values we established earlier
	with np.load('model84.npz') as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))]
	lasagne.layers.set_all_param_values(network, param_values)
	
	
	f = open("pickups.txt", 'r')
	text = f.read()
	lines = text.split("\n")
	random.seed(1)
	print("Pick up lines ready")

	session = pynder.Session(token)
	print("Session created")

	matches = session.matches()
	previous_match_num = len(matches)

	nearby_users = session.nearby_users() # returns a list of users nearby

	cv2.namedWindow("Image")
	for user in nearby_users:
	
		if AUTO == False:
			if raw_input("Press 0 to exit, any other key to continue") == "0":
				break
	
		previous_match_num = len(session.matches())	
		print("Current number of matches: " + str(previous_match_num))

		print("Name of the current user: " + user.name)
		photo_urls = user.get_photos(width=str(IMAGE_SIZE)) # a list of photo urls

		num_photos = len(photo_urls)
		print(num_photos)
	
	# list of numpy images to send to the AI
		face_rois = []
		for photo_url in photo_urls:
			if AUTO == False:			
				if raw_input("Press 0 to break out, or anything to continue") == '0':
					break

			response = requests.get(photo_url)
			image = Image.open(BytesIO(response.content))
#print(image)
			image_numpy = np.array(image)
#print(image_numpy)

# identify the face
			face_cascade = cv2.CascadeClassifier(path+'haarcascade_frontalface_alt.xml')
			image_gray = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2GRAY)

#faces = face_cascade.detectMultiScale(image_gray,1.1, 5, (30,30))
			faces = face_cascade.detectMultiScale(image_gray, scaleFactor=SCALE_FACTOR, minNeighbors=MIN_NEIGHBOURS, minSize=(MIN_SIZE, MIN_SIZE))
		#print("len of faces" + str(len(faces)))
		
			for (x,y,w,h) in faces:
				cv2.rectangle(image_numpy, (x, y), (x+w, y+h), (0, 255, 0), 2)
				roi = image_numpy[y:y+h , x:x+w]
				roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
				face_rois.append(roi_gray)
				break # just get one of the faces

			cv2.imshow("Image", image_numpy)
			cv2.waitKey(10)

		# send face_images to the NN
		# use trained function

		y_predicted = []
		for face_image in face_rois:
			# first resize it
			face_image_resized = cv2.resize(face_image, (PIXELS,PIXELS)) 
			cv2.imshow("face_image_resized", face_image_resized)
			x_input = np.zeros((1,1,PIXELS,PIXELS))
			x_input[0][0] = face_image_resized

			test_prediction_val = lasagne.layers.get_output(network, deterministic=True)
			predict_fn = theano.function([input_var], test_prediction_val)
			y = predict_fn(x_input)

			print(y)
			y_val = np.argmax(y, axis=1)
			print(y_val)
			
			y_predicted.extend(y_val)

		print(y_predicted[:])
		if (len(y_predicted) >0):
			score = sum(y_predicted)/len(y_predicted)
		else:
			score = 0
		
		print("Press l to like and p to pass and 0 to take no action")
		# input = raw_input()
		if score > THRESHOLD:
			action = "LIKE"	
			user.like()
		else:
			action = "PASS"
			user.dislike()


		cv2.putText(image_numpy,action,(int(IMAGE_SIZE*0.7),int(IMAGE_SIZE*0.3)),FONT,TEXT_SIZE,(0,0,0),LINE_THICKNESS)	
		cv2.imshow("Image", image_numpy)
	
	#check if a a match has been obtained
		matches = session.matches() # this function is pretty slow
		if len(matches) > previous_match_num:
		# then a match has occured so send em a message...
			if raw_input("A match has occured press 1 to send a message") == '1':
				pickup_line = lines[random.randint(0,len(lines))]
			# this line wouldnt work
				matches[len(matches)-1].message(pickup_line)
				print("Pick up line sent: " + pickup_line)		
	
		previous_match_num = len(matches)
	
		cv2.waitKey(10)

	cv2.destroyAllWindows() 


if __name__ == '__main__':
	main()
