import re
import os
import math
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearnex.svm import SVC


def read_image(file_name):
    with open(file_name, 'rb') as f:
        buf = f.read()
    header, width, height, max_val = re.search(
        b"(^P5\s(?:\s*#.*[\r\n])*(\d+)\s(?:\s*#.*[\r\n])*"
        b"(\d+)\s(?:\s*#.*[\r\n])*"
        b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buf).groups()
    return np.frombuffer(buf, dtype='u1' if int(max_val) < 256 else '>u2',
                         count=int(width) * int(height), offset=len(header)
                         ).reshape((int(height), int(width)))


def get_images():
    all_images = []
    images = os.listdir('../dataset/data1/')
    all_images += [i for i in images if i[-4:] == '.pgm']
    filtered = []
    for images in all_images:
        try:
            filtered.append([images, read_image('../dataset/data1/{}'.format(images))])
        except:
            continue
    return filtered


def fill_bin(x, y, mo, spaces, bn):
    for i in range(*y):
        for j in range(*x):
            orientation = mo[i][j][1]
            closest = 0
            val = abs(orientation - spaces[0])
            for s in range(1, len(spaces)):
                if abs(orientation - spaces[s]) < val:
                    closest = s
                    val = abs(orientation - spaces[s])
            bn[closest] += mo[i][j][0]


def compute_hog_features(image):
    gx, gy = np.gradient(image)
    mo = []
    for y in range(len(gx)):
        mo.append([])
        for x in range(len(gx[y])):
            mo[y].append([
                math.sqrt(math.pow(gx[y][x], 2) + math.pow(gy[y][x], 2)),
                math.atan2(gy[y][x], gx[y][x])])
    spaces = [-math.pi + ((math.pi * 2) / 12) * i for i in range(12)]
    csize = image.shape[0] // 3
    ycells = [(0, csize), (csize, csize * 2), (csize * 2, image.shape[0])]
    xcells = [(0, csize), (csize, csize * 2), (csize * 2, image.shape[1])]
    cbins = []
    for y in ycells:
        for x in xcells:
            cbins.append([0 for _ in range(12)])
            fill_bin(x, y, mo, spaces, cbins[-1])
    del mo
    cbins = [bn / np.linalg.norm(bn) for bn in cbins]
    return np.array(cbins).flatten()


def get_feature_labels(images):
    vals = ['megak', 'night', 'glickman', 'cheyer', 'an2i', 'bpm',
            'saavik', 'kk49', 'tammo', 'steffi', 'boland', 'mitchell',
            'sz24', 'danieln', 'karyadi', 'ch4f', 'kawamura', 'phoebe',
            'at33', 'choon']
    labels = [vals.index(i[0].split('_')[0]) for i in images]
    features = [compute_hog_features(i[1]) for i in images]
    return features, labels


def cross_validation(train, test, model):
    scores = cross_val_score(model, train, test, cv=10)
    return scores.mean(), scores.std()


if __name__ == "__main__":
    images = get_images()
    feature, label = get_feature_labels(images)
    xtrain, xtest, ytrain, ytest = train_test_split(feature, label,
                                                    test_size=.2)
    # clf = RandomForestClassifier(n_estimators=100, criterion='entropy')
    # clf = DecisionTreeClassifier(n_estimators=100, criterion='entropy')
    clf = SVC()
    clf.fit(xtrain, ytrain)
    predicted = clf.predict(xtest)
    print(accuracy_score(ytest, predicted))
    print(cross_validation(feature, label, clf))
