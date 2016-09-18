from bs4 import BeautifulSoup
import requests
import re
import urllib2
import os
import cookielib
import json

def get_soup(url,header):
    return BeautifulSoup(urllib2.urlopen(urllib2.Request(url,headers=header)),'html.parser')

folder = raw_input("Positive or Negative")
destination = "/home/henry/Documents/Tinder/Database/" + folder
 
query = raw_input("query image")# you can change the query for the image  here
image_type = "img"

query= query.split()
query='+'.join(query)
url="https://www.google.co.in/search?q="+query+"&source=lnms&tbm=isch"
print url
#add the directory for your image here

header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"
}
soup = get_soup(url,header)


ActualImages=[]# contains the link for Large original images, type of  image
for a in soup.find_all("div",{"class":"rg_meta"}):
    link , Type =json.loads(a.text)["ou"]  ,json.loads(a.text)["ity"]
    ActualImages.append((link,Type))

print  "there are total" , len(ActualImages),"images"


for i , (img , Type) in enumerate( ActualImages):
    try:
        req = urllib2.Request(img, headers={'User-Agent' : header})
        raw_img = urllib2.urlopen(req).read()

        cntr = len([i for i in os.listdir(destination) if image_type in i]) + 1
        print cntr
        if len(Type)==0:
            f = open(os.path.join(destination , image_type + "_"+ str(cntr)+".jpg"), 'wb')
        else :
            f = open(os.path.join(destination, image_type + "_"+ str(cntr)+"."+Type), 'wb')


        f.write(raw_img)
        f.close()
    except Exception as e:
        print "could not load : "+img
        print e




directory ="/home/henry/Documents/Tinder/Database/" + folder + "/"

image_counter = 1000
for i in range(0,1):
        for filename in os.listdir(directory):
                if filename.endswith("jpg"):
                        new_filename = str(image_counter) + ".jpg"
                        print(directory + filename)
                        os.rename(directory + filename, directory + new_filename)
                        # rename in a new directory
                        image_counter = image_counter + 1


