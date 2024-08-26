from tkinter import *
from tkinter import messagebox, colorchooser
import numpy as np
import cv2 as cv
from model import Model
from storage import Storage


class ImageClassifier:

    def __init__(self):
        """
        Initializes the ImageClassifier class
        """
        self.model = Model()
        self.storage = Storage()

        self.class1 = None
        self.class2 = None
        self.class3 = None

        self.class1_counter = 1
        self.class2_counter = 1
        self.class3_counter = 1

        self.model_name = None
        self.root = None
        self.drawing = None
        self.status_label = None
        self.canvas = None

        self.brush_width = 12
        self.drawing_color = (0, 0, 0)

        self.storage.load_project(self)
        self.init_gui()

    def init_gui(self):
        """
        Initializes the GUI
        """
        WIDTH = 550
        HEIGHT = 550
        BORDER_SIZE = 2

        button_style = {
            "font": ("Helvetica", 11, "bold"),
            "bg": "#008B8B",
            "fg": "white",
            "relief": "raised",
            "bd": 3,
            "activebackground": "#006161",
            "activeforeground": "white"
        }

        self.root = Tk()
        self.root.title("Image Classifier")

        canvas_frame = Frame(self.root, bg="black", padx=BORDER_SIZE, pady=BORDER_SIZE)
        canvas_frame.pack(expand=YES, fill=BOTH, padx=BORDER_SIZE + 2, pady=BORDER_SIZE + 2)
        self.canvas = Canvas(canvas_frame, width=WIDTH, height=HEIGHT, bg="white", bd=0)
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)

        self.drawing = np.ones((HEIGHT + 4, WIDTH + 4, 3), np.uint8) * 255

        btn_frame = Frame(self.root, bg="white")
        btn_frame.pack(fill=X, side=BOTTOM)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        btn_frame.columnconfigure(2, weight=1)

        Button(btn_frame, text=self.class1, command=lambda: self.storage.save_image(self, 1),
                            **button_style).grid(row=0, column=0, sticky=W + E, padx=5, pady=5)

        Button(btn_frame, text=self.class2, command=lambda: self.storage.save_image(self, 2),
                            **button_style).grid(row=0, column=1, sticky=W + E, padx=5, pady=5)

        Button(btn_frame, text=self.class3, command=lambda: self.storage.save_image(self, 3),
                            **button_style).grid(row=0, column=2, sticky=W + E, padx=5, pady=5)

        (Button(btn_frame, text="Train Model", command=self.train_model, **button_style).
            grid(row=1, column=0, sticky=W + E, padx=5, pady=5))
        (Button(btn_frame, text="Predict", command=self.predict, **button_style).
            grid(row=1, column=1, sticky=W + E, padx=5, pady=5))
        (Button(btn_frame, text="Save Model", command=self.save_everything, **button_style).
            grid(row=1, column=2, sticky=W + E, padx=5, pady=5))
        (Button(btn_frame, text="Choose Color", command=self.choose_color, **button_style).
            grid(row=2, column=0, sticky=W + E, padx=5, pady=5))
        (Button(btn_frame, text="Clear", command=self.clear_canvas, **button_style).
            grid(row=2, column=1, sticky=W + E, padx=5, pady=5))
        (Button(btn_frame, text="Show Accuracy", command=self.show_accuracy, **button_style).
         grid(row=2, column=2, sticky=W + E, padx=5, pady=5))

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.attributes("-topmost", True)
        self.root.mainloop()

    def paint(self, event):
        """
        Handles the drawing on the canvas when the mouse is moved while the button is held down.
        """
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        color_hex = '#{:02x}{:02x}{:02x}'.format(*self.drawing_color)

        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color_hex, outline=color_hex, width=self.brush_width)
        cv.rectangle(self.drawing, (x1, y1), (x2 + self.brush_width, y2 + self.brush_width), self.drawing_color, -1)

    def choose_color(self):
        """
        Opens a color chooser dialog to select a new drawing color and updates the current drawing color.
        """
        color = colorchooser.askcolor(title="Choose drawing color")
        if color[0]:
            self.drawing_color = tuple(int(c) for c in color[0])

    def clear_canvas(self):
        """
        Clears the canvas
        """
        self.canvas.delete("all")
        self.drawing.fill(255)

    def train_model(self):
        """
        Trains the model using the current drawings and classifications.
        """
        self.model.train_model(self)

    def predict(self):
        """
        Predicts the class of the current drawing using the trained model.
        """
        self.model.predict(self)

    def save_everything(self):
        """
        Saves the current state of the model, including the drawings and their classifications.
        """
        self.storage.save_model(self, self.model)

    def show_accuracy(self):
        """
        Displays a plot of the model's prediction accuracy over time.
        """
        self.model.show_accuracy()

    def on_closing(self):
        """
        Handles the event when the user attempts to close the application.
        Prompts the user to save their work before exiting.
        """
        answer = messagebox.askyesnocancel("Quit?", "Do you want to save your work?", parent=self.root)
        if answer is not None:
            if answer:
                self.save_everything()
            self.root.destroy()
            exit()


if __name__ == '__main__':
    ImageClassifier()
