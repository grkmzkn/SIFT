import argparse as ap
import cv2
import imutils 
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from scipy._lib.six import xrange
from matplotlib import pyplot as plt

clf, classes_names, stdSlr, k, voc = joblib.load("bof.pkl")

parser = ap.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-t", "--testingSet", help="Path to testing Set")
group.add_argument("-i", "--image", help="Path to image")
parser.add_argument('-v',"--visualize", action='store_true')
args = vars(parser.parse_args())

image_paths = []
if args["testingSet"]:
    test_path = args["testingSet"]
    try:
        testing_names = os.listdir(test_path)
    except OSError:
        print("No such directory {}\nCheck if the file exists".format(test_path))
        exit()
    for testing_name in testing_names:
        dir = os.path.join(test_path, testing_name)
        class_path = imutils.imlist(dir)
        image_paths+=class_path
else:
    image_paths = [args["image"]]
    
fea_det = cv2.xfeatures2d.SIFT_create()

des_list = []

for image_path in image_paths:
    im = cv2.imread(image_path)
    if im.any() == None:
        print("No such file {}\nCheck if the file exists".format(image_path))
        exit()
    kpts, des = fea_det.detectAndCompute(im, None)
    des_list.append((image_path, des))   
    
descriptors = des_list[0][1]
for image_path, descriptor in des_list[0:]:
    descriptors = np.vstack((descriptors, descriptor)) 

test_features = np.zeros((len(image_paths), k), "float32")
for i in xrange(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        test_features[i][w] += 1

nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

test_features = stdSlr.transform(test_features)

predictions =  [classes_names[i] for i in clf.predict(test_features)]

fig= plt.figure(figsize=(60, 100))
columns = 12
rows = 4
sayac = 1

if args["visualize"]:
    for image_path, prediction in zip(image_paths, predictions):
        image = cv2.imread(image_path)
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        pt = (0, 3 * image.shape[0] // 4)
        cv2.putText(image, prediction, pt ,cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 2)
        fig.add_subplot(rows, columns, sayac)
        cv2.imshow("Image", image)
        sayac+= 1
        #cv2.waitKey(3000)
plt.show()