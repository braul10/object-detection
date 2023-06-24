import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from transformers import YolosImageProcessor, YolosForObjectDetection
import torch

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Object detection")
        self.image = None
        self.image2 = None
        self.canvas_size = (600, 600)
        self.file_path = ""

        self.model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
        self.image_processor = YolosImageProcessor.from_pretrained('hustvl/yolos-tiny')

        self.select_button = tk.Button(root, text="Select image", command=self.load_image)
        self.select_button.pack()

        self.canvas = tk.Canvas(root, width=self.canvas_size[0], height=self.canvas_size[1], bg="black")
        self.canvas.pack(side='left')

        self.process_button = tk.Button(root, text="Process", command=self.process_image)
        self.process_button.pack(side='left')

        self.canvas2 = tk.Canvas(root, width=self.canvas_size[0], height=self.canvas_size[1], bg="black")
        self.canvas2.pack(side='left')
        self.colors = [
            "Red",
            "Green",
            "Blue",
            "Yellow",
            "Orange",
            "Pink",
            "Purple",
            "Brown",
            "Gray",
            "Cyan",
            "Magenta",
            "Turquoise",
            "Lime",
            "SkyBlue",
            "Beige",
            "Gold",
            "Silver",
            "Coral",
            "White",
            "Black"
        ];


    def load_image(self):
        self.canvas.delete('all')
        self.canvas2.delete('all')
        self.file_path = filedialog.askopenfilename()
        image = Image.open(self.file_path)
        image.thumbnail(self.canvas_size, Image.Resampling.LANCZOS)  # Aquí redimensionamos la imagen
        self.image = ImageTk.PhotoImage(image)

        # Calculamos la posición para centrar la imagen
        x = (self.canvas_size[0] - self.image.width()) / 2
        y = (self.canvas_size[1] - self.image.height()) / 2
        self.canvas.create_image(x, y, anchor='nw', image=self.image)

    def process_image(self):
        if self.image is None:
            return

        original_image = Image.open(self.file_path)
        inputs = self.image_processor(images=original_image, return_tensors="pt")
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([original_image.size[::-1]])
        results = self.image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

        self.image2 = self.image
        # Calculamos la posición para centrar la imagen
        x_offset = (self.canvas_size[0] - self.image2.width()) / 2
        y_offset = (self.canvas_size[1] - self.image2.height()) / 2
        self.canvas2.create_image(x_offset, y_offset, anchor='nw', image=self.image2)

        # Calculamos los factores de escala
        scale_x = self.image2.width() / original_image.width
        scale_y = self.image2.height() / original_image.height

        # Dibujamos los rectángulos de detección de objetos
        count = 0
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            print(f"{self.model.config.id2label[label.item()]} ({round(score.item(), 3)}) -> {box}")
            self.canvas2.create_rectangle(box[0]*scale_x + x_offset,
                                          box[1]*scale_y + y_offset,
                                          box[2]*scale_x + x_offset,
                                          box[3]*scale_y + y_offset,
                                          outline=self.colors[count % len(self.colors)])
            label_name = self.model.config.id2label[label.item()]
            self.canvas2.create_text(box[0]*scale_x + x_offset,
                                    box[1]*scale_y + y_offset - 17,
                                    anchor='nw',
                                    text=label_name,
                                    fill=self.colors[count % len(self.colors)])
            count += 1

root = tk.Tk()
app = App(root)
root.mainloop()
