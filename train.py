from PIL import Image
from feature import NPDFeature
from ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.metrics import classification_report
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

    clf = AdaBoostClassifier(tree.DecisionTreeClassifier, 10)

    #cal NPD
    for i in range(num_face+num_nonface):
        NPD.append(NPDFeature(imgs[i]).extract())

    #save NPD
    # clf.save(NPD,'output')


    #load NPD
    # NPD=np.array(clf.load('output'))

    train_x=np.row_stack((NPD[0:100],NPD[500:600]))
    train_y=np.append(np.ones((1,100)),np.linspace(-1,-1,100))
    test_x=np.row_stack((NPD[200:300],NPD[700:800]))
    test_y = np.append(np.ones((1, 100)), np.linspace(-1, -1, 100))

    clf.fit(train_x, train_y)
    y=clf.predict(test_x)

    hit=0
    for i in range(test_x.shape[0]):
        if (y[i]==test_y[i]):
            hit+=1
    print("hit rate:",hit/test_x.shape[0])

    with open('report.txt','w') as f:
        f.write(classification_report(test_y, y))





