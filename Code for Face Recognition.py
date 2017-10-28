import cv2
import numpy as np

# Initialize video capture object
cap = cv2.VideoCapture(0)
# Define the image size scaling factor
scaling_factor = 0.5
# Loop until you hit the Esc key
while True:
    # Capture the current frame
    ret, frame = cap.read()
	# Resize the frame
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, 
            interpolation=cv2.INTER_AREA)
	# Display the image
    cv2.imshow('Webcam', frame)
	# Detect if the Esc key has been pressed
    c = cv2.waitKey(1)
    if c == 27:
        break
# Release the video capture object
cap.release()
# Close all active windows
cv2.destroyAllWindows()
 
# Load the face cascade file
face_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_frontalface_alt.xml')
# Check if the face cascade file has been loaded
if face_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')
# Initialize the video capture object
cap = cv2.VideoCapture(0)
# Define the scaling factor
scaling_factor = 0.5
# Loop until you hit the Esc key
while True:
    # Capture the current frame and resize it
    ret, frame = cap.read()
				    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
	# Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Run the face detector on the grayscale image
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
	# Draw rectangles on the image
    for (x,y,w,h) in face_rects:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
	# Display the image
    cv2.imshow('Face Detector', frame)
	# Check if Esc key has been pressed
    c = cv2.waitKey(1)
    if c == 27:
        break
# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

# Load face, eye, and nose cascade files
face_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_mcs_nose.xml')
# Check if face cascade file has been loaded
if face_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')
# Check if eye cascade file has been loaded
if eye_cascade.empty():
    raise IOError('Unable to load the eye cascade classifier xml file')
# Check if nose cascade file has been loaded
if nose_cascade.empty():
    raise IOError('Unable to load the nose cascade classifier xml file')
# Initialize video capture object and define scaling factor
cap = cv2.VideoCapture(0)
	scaling_factor = 0.5
	while True:
    # Read current frame, resize it, and convert it to grayscale
        ret, frame = cap.read()
		frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Run face detector on the grayscale image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
						    # Run eye and nose detectors within each face rectangle
    for (x,y,w,h) in faces:
						        # Grab the current ROI in both color and grayscale images
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
						        # Run eye detector in the grayscale ROI
        eye_rects = eye_cascade.detectMultiScale(roi_gray)
						        # Run nose detector in the grayscale ROI
        nose_rects = nose_cascade.detectMultiScale(roi_gray, 1.3, 5)
						        # Draw green circles around the eyes
        for (x_eye, y_eye, w_eye, h_eye) in eye_rects:
            center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye))
            radius = int(0.3 * (w_eye + h_eye))
            color = (0, 255, 0)
            thickness = 3
            cv2.circle(roi_color, center, radius, color, thickness)
						        for (x_nose, y_nose, w_nose, h_nose) in nose_rects:
            cv2.rectangle(roi_color, (x_nose, y_nose), (x_nose+w_nose, 
                y_nose+h_nose), (0,255,0), 3)
            break
						    # Display the image
    cv2.imshow('Eye and nose detector', frame)
						    # Check if Esc key has been pressed
    c = cv2.waitKey(1)
    if c == 27:
        break
						# Release video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
						import numpy as np
from sklearn import decomposition 
								# Define individual features
x1 = np.random.normal(size=250)
x2 = np.random.normal(size=250)
x3 = 2*x1 + 3*x2
x4 = 4*x1 - x2
x5 = x3 + 2*x4
								# Create dataset with the above features
X = np.c_[x1, x3, x2, x5, x4]
								# Perform Principal Components Analysis
pca = decomposition.PCA()
								pca.fit(X)
								# Print variances
variances = pca.explained_variance_
print '\nVariances in decreasing order:\n', variances
								# Find the number of useful dimensions
thresh_variance = 0.8
num_useful_dims = len(np.where(variances > thresh_variance)[0])
print '\nNumber of useful dimensions:', num_useful_dims
								# As we can see, only the 2 first components are useful
pca.n_components = num_useful_dims
								X_new = pca.fit_transform(X)
print '\nShape before:', X.shape
print 'Shape after:', X_new.shape
								import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles
										# Set the seed for random number generator
np.random.seed(7)
										# Generate samples
X, y = make_circles(n_samples=500, factor=0.2, noise=0.04)
										# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X)
										# Perform Kernel PCA
kernel_pca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
X_kernel_pca = kernel_pca.fit_transform(X)
X_inverse = kernel_pca.inverse_transform(X_kernel_pca)
										# Plot original data
class_0 = np.where(y == 0)
class_1 = np.where(y == 1)
plt.figure()
plt.title("Original data")
plt.plot(X[class_0, 0], X[class_0, 1], "ko", mfc='none')
plt.plot(X[class_1, 0], X[class_1, 1], "kx")
plt.xlabel("1st dimension")
plt.ylabel("2nd dimension")
										# Plot PCA projection of the data
plt.figure()
plt.plot(X_pca[class_0, 0], X_pca[class_0, 1], "ko", mfc='none')
plt.plot(X_pca[class_1, 0], X_pca[class_1, 1], "kx")
plt.title("Data transformed using PCA")
plt.xlabel("1st principal component")
plt.ylabel("2nd principal component")
										# Plot Kernel PCA projection of the data
plt.figure()
plt.plot(X_kernel_pca[class_0, 0], X_kernel_pca[class_0, 1], "ko", mfc='none')
plt.plot(X_kernel_pca[class_1, 0], X_kernel_pca[class_1, 1], "kx")
plt.title("Data transformed using Kernel PCA")
plt.xlabel("1st principal component")
plt.ylabel("2nd principal component")
										# Transform the data back to original space
plt.figure()
plt.plot(X_inverse[class_0, 0], X_inverse[class_0, 1], "ko", mfc='none')
plt.plot(X_inverse[class_1, 0], X_inverse[class_1, 1], "kx")
plt.title("Inverse transform")
plt.xlabel("1st dimension")
plt.ylabel("2nd dimension")
plt.show()
										import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import PCA, FastICA 
												# Load data
input_file = 'mixture_of_signals.txt'
X = np.loadtxt(input_file)
												# Compute ICA
ica = FastICA(n_components=4)
												# Reconstruct the signals
signals_ica = ica.fit_transform(X)
												# Get estimated mixing matrix
mixing_mat = ica.mixing_  
												# Perform PCA 
pca = PCA(n_components=4)
signals_pca = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components
												# Specify parameters for output plots 
models = [X, signals_ica, signals_pca]
												colors = ['blue', 'red', 'black', 'green']
												# Plotting input signal
plt.figure()
plt.title('Input signal (mixture)')
for i, (sig, color) in enumerate(zip(X.T, colors), 1):
    plt.plot(sig, color=color)
												# Plotting ICA signals 
plt.figure()
plt.title('ICA separated signals')
plt.subplots_adjust(left=0.1, bottom=0.05, right=0.94, 
        top=0.94, wspace=0.25, hspace=0.45)
												for i, (sig, color) in enumerate(zip(signals_ica.T, colors), 1):
    plt.subplot(4, 1, i)
    plt.title('Signal ' + str(i))
    plt.plot(sig, color=color)
												# Plotting PCA signals  
plt.figure()
plt.title('PCA separated signals')
plt.subplots_adjust(left=0.1, bottom=0.05, right=0.94, 
        top=0.94, wspace=0.25, hspace=0.45)
												for i, (sig, color) in enumerate(zip(signals_pca.T, colors), 1):
    plt.subplot(4, 1, i)
    plt.title('Signal ' + str(i))
    plt.plot(sig, color=color)
plt.show()
												import os
import cv2
import numpy as np
from sklearn import preprocessing 
														# Class to handle tasks related to label encoding
class LabelEncoder(object):
														    # Method to encode labels from words to numbers
    def encode_labels(self, label_words):
        self.le = preprocessing.LabelEncoder()
        self.le.fit(label_words)
														    # Convert input label from word to number
    def word_to_num(self, label_word):
        return int(self.le.transform([label_word])[0])
														    # Convert input label from number to word
    def num_to_word(self, label_num):
        return self.le.inverse_transform([label_num])[0]
														# Extract images and labels from input path
def get_images_and_labels(input_path):
    label_words = []
														    # Iterate through the input path and append files
    for root, dirs, files in os.walk(input_path):
        for filename in (x for x in files if x.endswith('.jpg')):
            filepath = os.path.join(root, filename)
            label_words.append(filepath.split('/')[-2]) 
														    # Initialize variables
    images = []
    le = LabelEncoder()
    le.encode_labels(label_words)
    labels = []
														    # Parse the input directory
    for root, dirs, files in os.walk(input_path):
        for filename in (x for x in files if x.endswith('.jpg')):
            filepath = os.path.join(root, filename)
														            # Read the image in grayscale format
            image = cv2.imread(filepath, 0) 
														            # Extract the label
            name = filepath.split('/')[-2]
														            # Perform face detection
            faces = faceCascade.detectMultiScale(image, 1.1, 2, minSize=(100,100))
														            # Iterate through face rectangles
            for (x, y, w, h) in faces:
                images.append(image[y:y+h, x:x+w])
                labels.append(le.word_to_num(name))
    return images, labels, le
														if __name__=='__main__':
    cascade_path = "cascade_files/haarcascade_frontalface_alt.xml"
    path_train = 'faces_dataset/train'
    path_test = 'faces_dataset/test'
														    # Load face cascade file
    faceCascade = cv2.CascadeClassifier(cascade_path)
														    # Initialize Local Binary Patterns Histogram face recognizer
    recognizer = cv2.face.createLBPHFaceRecognizer()
														    # Extract images, labels, and label encoder from training dataset
    images, labels, le = get_images_and_labels(path_train)
														    # Train the face recognizer 
    print "\nTraining..."
    recognizer.train(images, np.array(labels))
														    # Test the recognizer on unknown images
    print '\nPerforming prediction on test images...'
    stop_flag = False
    for root, dirs, files in os.walk(path_test):
        for filename in (x for x in files if x.endswith('.jpg')):
            filepath = os.path.join(root, filename)
														            # Read the image
            predict_image = cv2.imread(filepath, 0)
														            # Detect faces
            faces = faceCascade.detectMultiScale(predict_image, 1.1, 
                    2, minSize=(100,100))
														            # Iterate through face rectangles
            for (x, y, w, h) in faces:
                # Predict the output
                predicted_index, conf = recognizer.predict(
                        predict_image[y:y+h, x:x+w])
														                # Convert to word label
                predicted_person = le.num_to_word(predicted_index)
														                # Overlay text on the output image and display it
                cv2.putText(predict_image, 'Prediction: ' + predicted_person, 
                        (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 6)
                cv2.imshow("Recognizing face", predict_image)
														            c = cv2.waitKey(0)
            if c == 27:
                stop_flag = True
                break
        if stop_flag:
            break
														