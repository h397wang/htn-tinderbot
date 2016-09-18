import random
import pynder
import requests
import numpy as np
import cv2
from PIL import Image
#from StringIO import StringIO
from io import BytesIO

# get the token from here
# https://www.facebook.com/dialog/oauth?client_id=464891386855067&redirect_uri=fbconnect%3A%2F%2Fsuccess&scope=basic_info%2Cemail%2Cpublic_profile%2Cuser_about_me%2Cuser_activities%2Cuser_birthday%2Cuser_education_history%2Cuser_friends%2Cuser_interests%2Cuser_likes%2Cuser_location%2Cuser_photos%2Cuser_relationship_details&response_type=token&__mref=message_bubble

AUTO = False

token = "EAAGm0PX4ZCpsBAMW4AdXJTS1U6BGtL6PV0DXw0322TgmZAtJwqwxw5wtK2Im3HSgwtyPahpSu3XDx1A8LpJyS0vnZBQbZCJvasPpVPDAPcilFXfgVqiQBRf9dS3IspYZC3DeO3fRqZCbGoDxsMohwi5ryxmvWTmullJKbbxfB2zAZDZD"
id = "591469779"

path = "/home/henry/opencv/data/haarcascades/"
# img from url request
IMAGE_SIZE = 640

# detectMultiScale parameters
MIN_NEIGHBOURS = 6
MIN_SIZE = 30

# printing text parameters
LINE_THICKNESS = 15
TEXT_SIZE = 5

f = open("pickups.txt", 'r')
text = f.read()
lines = text.split("\n")
random.seed(1)

session = pynder.Session(token)
print("Session created")

matches = session.matches()
previous_match_num = len(matches)

nearby_users = session.nearby_users() # returns a list of users nearby

#print(type(nearby_users[0]))
#print(type(matches[0]))

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
	face_images = []
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
		faces = face_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=MIN_NEIGHBOURS, minSize=(MIN_SIZE, MIN_SIZE))
		#print("len of faces" + str(len(faces)))
		
		for (x,y,w,h) in faces:
			cv2.rectangle(image_numpy, (x, y), (x+w, y+h), (0, 255, 0), 2)
			roi = image_numpy[y:y+h , x:x+w]
			roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
			face_images.append(roi_gray)

		#cv2.namedWindow("Image")		
		cv2.imshow("Image", image_numpy)
		cv2.waitKey(10)
#		cv2.destroyAllWindows()
	# send face_images to the NN
	# use trained function
#	for face in face_images:
	# for now simulate with key inputs lol
	print("Press l to like and p to pass and 0 to take no action")
	input = raw_input()
	if input == 'l':
		action = "LIKE"	
		user.like()
	elif input == 'p':
		action = "PASS"
		user.dislike()
	else:
		break

	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(image_numpy,action,(int(IMAGE_SIZE/2),int(IMAGE_SIZE/2)),font,TEXT_SIZE,(0,0,0),LINE_THICKNESS)	
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

cv2.closeAllWindows() 
