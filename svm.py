import os
import pickle
import cv2 as cv
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def data_preprocess(image_file):
    img = cv.imread(image_file)
    img = cv.resize(img, (256, 256), interpolation=cv.INTER_CUBIC)
    img = img.flatten()
    img = img / np.mean(img)
    return img
   
def load_data(image_dir="dataset/stair/public"):
    image_list = []
    label_list = []
    for label in os.listdir(image_dir):
        if label == 'no_stairs':
            images = os.listdir(os.path.join(image_dir, label))
            for image_file in images:
                label_list.append(label)
                image_list.append(data_preprocess(os.path.join(image_dir, label, image_file)))
        else:
            for direction in ['up', 'down']:
                images = os.listdir(os.path.join(image_dir, label, direction))
                for image_file in images:
                    label_list.append(label)
                    image_list.append(data_preprocess(os.path.join(image_dir, label, direction, image_file)))
    return np.asarray(image_list), np.asarray(label_list)


if __name__ == "__main__":
    
    image_array, label_array = load_data()
    image_train, image_test, label_train, label_test = train_test_split(image_array, label_array, test_size=0.2, random_state=42)
    
    print("Train stage")
    if os.path.isfile("pretrained_models/svm_model.pkl"):
        svm = pickle.load(open("pretrained_models/svm_model.pkl", "rb"))
    else:
        svm = SVC(kernel='rbf',gamma=0.001)
        svm.fit(image_train, label_train)
        pickle.dump(svm, open("pretrained_models/svm_model.pkl", "wb"))

    print("Test stage")
    count = 0
    acc_pred = 0
    for image, label_true in zip(image_test, label_test):
        image = image.reshape(1, -1)
        label_pred = svm.predict(image)[0]
        if label_true == label_pred:
            acc_pred += 1
        count += 1
    accuracy = float(acc_pred) / float(count) * 100

    print (f"Accuracy:{accuracy}%")
