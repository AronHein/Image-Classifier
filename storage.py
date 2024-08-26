import pickle
import os
from tkinter import simpledialog, messagebox
import random
import cv2 as cv


class Storage:
    def load_project(self, gui):
        """
        Loads the project settings and directories, or initializes a new project.
        Prompts the user for the project name and class names.
        """

        answer = messagebox.askyesnocancel("Load Project?", "Do you want to load a saved project?")
        if answer:
            self.load_saved_project(gui)
        elif not answer:
            self.load_new_project(gui)
        elif answer is None:
            gui.root.destroy()

    def load_saved_project(self, gui):
        """
        Loads a saved project
        """
        gui.model_name = simpledialog.askstring("Model Name", "Please enter your saved model name!", parent=gui.root)
        if os.path.exists(gui.model_name):
            self.load_model(gui)
        else:
            messagebox.showerror("Error", "The Model name you entered does not exist. Please try again.",
                                 parent=gui.root)
            self.load_saved_project(gui)

    def load_new_project(self, gui):
        """
        Loads a new project and prompts the user for the class names.
        """
        gui.model_name = simpledialog.askstring("Model Name", "Please enter your new model name!", parent=gui.root)
        answer = messagebox.askyesno("Random Classes?", "Do you want to get random classes?")
        if answer:
            classes = self.get_random_classes()
            gui.class1 = classes[0]
            gui.class2 = classes[1]
            gui.class3 = classes[2]
        else:
            gui.class1 = simpledialog.askstring("Class 1", "What is the name of the first class?", parent=gui.root)
            gui.class2 = simpledialog.askstring("Class 2", "What is the name of the second class?", parent=gui.root)
            gui.class3 = simpledialog.askstring("Class 3", "What is the name of the third class?", parent=gui.root)

        os.mkdir(gui.model_name)
        os.chdir(gui.model_name)
        os.mkdir(gui.class1)
        os.mkdir(gui.class2)
        os.mkdir(gui.class3)
        os.chdir("..")

    def get_random_classes(self):
        """
        Gets 3 random classes from classes.txt
        """
        with open('classes.txt', 'r') as file:
            classes = [line.strip() for line in file]
        random_classes = []
        while len(random_classes) < 3:
            random_class = random.choice(classes)
            if random_class not in random_classes:
                random_classes.append(random_class)

        return random_classes

    def save_model(self, gui, model):
        """
        Saves the current state of the model, including the drawings and their classifications.
        """
        data = {
            "c1": gui.class1,
            "c2": gui.class2,
            "c3": gui.class3,
            "c1c": gui.class1_counter,
            "c2c": gui.class2_counter,
            "c3c": gui.class3_counter,
            "clf": gui.model.clf,
            "mname": gui.model_name,
            "history": model.prediction_history,
            "correct": model.correct_predictions,
            "incorrect": model.incorrect_predictions
        }
        with open(f"{gui.model_name}/{gui.model_name}_data.pickle", "wb") as f:
            pickle.dump(data, f)
        messagebox.showinfo("Image Classifier", "Model successfully saved!", parent=gui.root)

    def load_model(self, gui):
        """
        Loads the current state of the model, including the drawings and their classifications.
        """
        with open(f"{gui.model_name}/{gui.model_name}_data.pickle", "rb") as f:
            data = pickle.load(f)
        gui.class1 = data['c1']
        gui.class2 = data['c2']
        gui.class3 = data['c3']
        gui.class1_counter = data['c1c']
        gui.class2_counter = data['c2c']
        gui.class3_counter = data['c3c']
        gui.model.clf = data['clf']
        gui.model_name = data['mname']
        gui.prediction_history = data['history']
        gui.correct_predictions = data['correct']
        gui.incorrect_predictions = data['incorrect']

    def save_image(self, gui, class_num):
        """
        Saves a new drawn image and clears the canvas
        """
        img = cv.resize(gui.drawing, (50, 50), interpolation=cv.INTER_AREA)

        if class_num == 1:
            cv.imwrite(f"{gui.model_name}/{gui.class1}/{gui.class1_counter}.png", img)
            gui.class1_counter += 1
        elif class_num == 2:
            cv.imwrite(f"{gui.model_name}/{gui.class2}/{gui.class2_counter}.png", img)
            gui.class2_counter += 1
        elif class_num == 3:
            cv.imwrite(f"{gui.model_name}/{gui.class3}/{gui.class3_counter}.png", img)
            gui.class3_counter += 1

        gui.clear_canvas()
