import pynder
import os
import random

f = open("pickups.txt", 'r')
text = f.read()
lines = text.split("\n")

random.seed(1)


# returns a string
def send_line(session):
	rng = random.randin(0, len(lines))
	session.
	return

