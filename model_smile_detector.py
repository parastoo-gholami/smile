import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn import svm
from sklearn.model_selection import train_test_split
import pickle
def HOG_(image):
    hog_feature = hog(image, orientations=30, pixels_per_cell=(6, 6), cells_per_block=(1, 1), block_norm='L2-Hys',transform_sqrt=True)
    return hog_feature

def LBP_(image):
    lbp_feature=local_binary_pattern(image,8,1)
    hist_lbp, _ = np.histogram(lbp_feature, bins=int(lbp_feature.max() + 1), range=(0,int(lbp_feature.max() + 1)),density=True)
    return hist_lbp

def GLCM_(image):
    glcm = np.zeros((256, 256), dtype=np.uint8)
    height, width = image.shape
    for i in range(height - 1):
        for j in range(width - 1):
            p = image[i, j]
            q = image[i, j + 1]
            glcm[p, q] += 1
    glcm = glcm.astype(np.float64)
    glcm /= np.sum(glcm)
    contrast = np.sum(np.square(glcm - np.mean(glcm)))
    dissimilarity = np.sum(np.abs(glcm - np.mean(glcm)))
    homogeneity = np.sum(glcm / (1.0 + np.abs(i - j)))
    energy = np.sum(np.square(glcm))
    correlation = np.sum(np.divide((i - np.mean(glcm)) * (j - np.mean(glcm)), np.sqrt(np.sum(np.square(i - np.mean(glcm)))) * np.sqrt(np.sum(np.square(j - np.mean(glcm))))))
    glcm_feature = np.array([contrast, dissimilarity, homogeneity, energy, correlation])
    return glcm_feature
name = open("C:\\Users\\ASUS\\Desktop\\GENKI-R2009a\\Subsets\\GENKI-4K\\GENKI-4K_Images.txt", "r")
images = name.read().splitlines()
name.close()
HOG_features = []
LBP_features = []
GLCM_features = []
for image_path in images:
    image = cv2.imread("C:\\Users\\ASUS\\Desktop\\GENKI-R2009a\\files\\{}".format(image_path))
    image = cv2.resize(image,(100,100),interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_feature = HOG_(gray)
    HOG_features.append(hog_feature)
    lbp_feature = LBP_(gray)
    LBP_features.append(lbp_feature)
    glcm_feature = GLCM_(gray)
    GLCM_features.append(glcm_feature)
features=np.concatenate((LBP_features,HOG_features,GLCM_features),axis=1)
label = open("C:\\Users\\ASUS\\Desktop\\GENKI-R2009a\\Subsets\\GENKI-4K\\GENKI-4K_Labels.txt", "r")
read_label =label.read().splitlines()
label.close()
lable_0=[row[0] for row in read_label]
labels = [eval(i) for i in lable_0]
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=32)

clf = svm.SVC()
clf.fit(features_train, labels_train)

accuracy = clf.score(features_test, labels_test)
print("Accuracy:", accuracy)
with open('model.pkl','wb') as f:
    pickle.dump(clf,f)