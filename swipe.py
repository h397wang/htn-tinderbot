# Pynder Imports
import pynder
import urllib
import requests

token = "EAAGm0PX4ZCpsBAM2uyxg6P4fhnEnr2zvwnp7uZAQQ7qeoGDcjG8xZBz2I6Yj5vTIpSZA13tXQlxCJJlcoezZAWvEABvZBSUSLzDcLt0jyRjMehmtqCHEr11pB58CJXnTvApHi6iZAU7P7J5GWm9Ki2KDyZAk6g80eeKXRhk8xGNZBlwZDZD"
id = "591469779"
session = pynder.Session(id, token)
users = session.nearby_users()
for user in users:
	user.dislike()
	print('Disliked:', user.name)