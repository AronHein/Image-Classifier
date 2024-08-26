import cv2 as cv
import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from tkinter import simpledialog, messagebox


class Model:

    def __init__(self):
        """
        Initializes the model
        """
        self.clf = LinearSVC()

        self.correct_predictions = 0
        self.incorrect_predictions = 0
        self.prediction_history = []

    def train_model(self, gui):
        """
        Trains the model using the current drawings and classifications.
        The images are loaded, flattened, and used to fit the model.
        """
        img_list = []
        class_list = []

        for x in range(1, gui.class1_counter):
            img = cv.imread(f"{gui.model_name}/{gui.class1}/{x}.png", cv.IMREAD_GRAYSCALE)
            img = img.flatten()
            img_list.append(img)
            class_list.append(1)

        for x in range(1, gui.class2_counter):
            img = cv.imread(f"{gui.model_name}/{gui.class2}/{x}.png", cv.IMREAD_GRAYSCALE)
            img = img.flatten()
            img_list.append(img)
            class_list.append(2)

        for x in range(1, gui.class3_counter):
            img = cv.imread(f"{gui.model_name}/{gui.class3}/{x}.png", cv.IMREAD_GRAYSCALE)
            img = img.flatten()
            img_list.append(img)
            class_list.append(3)

        img_list = np.array(img_list)
        class_list = np.array(class_list)

        self.clf.fit(img_list, class_list)
        messagebox.showinfo("Drawing Classifier", "Model successfully trained!", parent=gui.root)

    def predict(self, gui):
        """
        Predicts the class of the current drawing using the trained model.
        Resizes and processes the drawing, then uses the model to predict its class.
        """
        img = cv.resize(gui.drawing, (50, 50), interpolation=cv.INTER_AREA)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = img.flatten()
        prediction = self.clf.predict([img])
        predicted_class = None

        if prediction[0] == 1:
            predicted_class = gui.class1
        elif prediction[0] == 2:
            predicted_class = gui.class2
        elif prediction[0] == 3:
            predicted_class = gui.class3

        self.give_feedback(gui, predicted_class)

    def give_feedback(self, gui, predicted_class):
        """
        Gives feedback based on the user's input after a prediction is made.
        If the prediction is correct, the drawing is saved under the predicted class.
        If incorrect, the user is prompted to provide the correct class, and the drawing is saved accordingly.
        """
        answer = messagebox.askyesno("Prediction Feedback",
                                             f"The drawing is probably a {predicted_class}. Is this correct?",
                                             parent=gui.root)
        if answer:
            if predicted_class == gui.class1:
                gui.storage.save_image(gui, 1)
            elif predicted_class == gui.class2:
                gui.storage.save_image(gui, 2)
            elif predicted_class == gui.class3:
                gui.storage.save_image(gui, 3)
            self.correct_predictions += 1

        if not answer:
            correct_class = simpledialog.askstring("Correct Class", "What is the correct class for this drawing?",
                                                   parent=gui.root).lower()
            if correct_class == gui.class1.lower():
                gui.storage.save_image(gui, 1)
            elif correct_class == gui.class2.lower():
                gui.storage.save_image(gui, 2)
            elif correct_class == gui.class3.lower():
                gui.storage.save_image(gui, 3)
            else:
                messagebox.showerror("Error", "The class name you entered does not exist. Please try again.",
                                     parent=gui.root)
            self.incorrect_predictions += 1

        self.update_accuracy()

    def show_accuracy(self):
        """
        Displays a plot of the model's prediction accuracy over time.
        """
        plt.plot(self.prediction_history, marker='o')
        plt.title('Model Prediction Accuracy Over Time')
        plt.xlabel('Number of Predictions')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.05)
        plt.grid(True)
        plt.show()

    def update_accuracy(self):
        """
        Updates the accuracy of the model's prediction history.
        """
        total_predictions = self.correct_predictions + self.incorrect_predictions
        if total_predictions > 0:
            accuracy = self.correct_predictions / total_predictions
        else:
            accuracy = 0
        self.prediction_history.append(accuracy)
