from PIL import Image
from feature import NPDFeature
from ensemble import AdaBoostClassifier
from sklearn import tree
import pickle
import os
import numpy as np

if __name__ == "__main__":
    # write your code here

    face_filename=os.listdir(r"datasets/original/face")
    face_filename.sort()
    nonface_filename=os.listdir(r"datasets/original/nonface")
    nonface_filename.sort()


    imgs=[]
    NPD=[]

    num_face=500
    num_nonface=500
    #face
    for i in range(num_face):
        im=Image.open('datasets/original/face/'+face_filename[i]).convert('L').resize((24, 24))
        imgs.append(np.array(im))
        # train_y.append(1)

    #nonface
    for i in range(num_nonface):
        im=Image.open('datasets/original/nonface/'+nonface_filename[i]).convert('L').resize((24, 24))
        imgs.append(np.array(im))
        # train_y.append(-1)

    clf = AdaBoostClassifier(tree.DecisionTreeClassifier, 3)

    #cal NPD && save

     for i in range(num_face+num_nonface):
         NPD.append(NPDFeature(imgs[i]).extract())
     clf.save(NPD,'output')


    # load NPD
    NPD=np.array(clf.load('output'))

    train_x=np.row_stack((NPD[0:100],NPD[500:600]))
    train_y=np.append(np.ones((1,100)),np.linspace(-1,-1,100))
    test_x=np.row_stack((NPD[200:300],NPD[700:800]))
    test_y = np.append(np.ones((1, 100)), np.linspace(-1, -1, 100))
    clf.fit(train_x, train_y)
    p=clf.predict(test_x)
    print(p)








