# Importing Modules
import numpy as np, cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import PIL.ImageOps

# Importing Libraries
from PIL import Image
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Fetch the data locally
x = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")['labels']
print(pd.Series(y).value_counts())
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nclasses = len(classes)

# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=7500, test_size=2500, random_state=9)
X_train_scale = X_train / 255.0
X_test_scale = X_test / 255.0

# Fitting the data to the model
clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scale, y_train)

# Calculating the accuracy of the model
y_pred = clf.predict(X_test_scale)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ', accuracy)

# Starting the camera
cap = cv2.VideoCapture(0)
while (True):
    try:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Drawing a box in the center of the video
        height, width = gray.shape
        upper_left = (int(width/2 - 56), int(height/2 - 56))
        lower_right = (int(width/2 + 56), int(height/2 + 56))
        cv2.rectangle(gray, upper_left, lower_right, (0, 255, 0), 2)
        # Do only consider the area inside the box for detecting the digit
        roi = gray[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]
        # Converting cv2 image to PIL image
        im_pil = Image.fromarray(roi)
        # Converting PIL image to grayscale - 'L' Format means each pixel is represented by a single integer value from 0 to 255
        image_bw = im_pil.convert('L')
        # Resizing the image to 28x28 pixels
        image_bw_resized = image_bw.resize((28, 28), Image.ANTIALIAS)
        # Inverting the colors of the image
        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
        pixel_filter = 20
        min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
        image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255)
        max_pixel = np.max(image_bw_resized_inverted)
        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
        test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1, 784)
        # Predicting the digit
        test_pred = clf.predict(test_sample)
        # Displaying the predicted digit
        print('Predicted digit: ', test_pred[0])
        # Displaying the resulting frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()