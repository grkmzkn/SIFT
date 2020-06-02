from PIL import Image
import os, sys

path = "C:\\Users\\grkmzkn\\Desktop\\boyut\\130\\"
dirs = os.listdir(path)

def resize():
	i=0
	for item in dirs:
		print (str(path+item))
		img = Image.open(path+item)
		f, e = os.path.splitext(path+item)
		imR = img.resize((60,100), Image.ANTIALIAS)
		imR.save("C:\\Users\\grkmzkn\\Desktop\\yeni\\130\\"+str("%02d" % (1,))+str("%03d" % (1,))+str("%06d" % (i))+ '.jpg', 'JPEG', quality=90, optimize=True, progressive=True)
		i=i+1
resize()