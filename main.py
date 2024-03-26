import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import cvzone
from ultralytics import YOLO
import math


## code for creating gui of start window
class ObjectDetectionApp:
    def __init__(self, root):  # sort of constructor
        self.root = root
        self.root.title("Object Detection App")
        self.root.configure(bg='#C6C8EE')
        # Variables
        self.video_path = ""
        self.model = YOLO("models/best.pt")
        self.class_names = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
                            'Safety Vest', 'machinery', 'vehicle']
        self.my_color = (0, 0, 255)
        # Create GUI elements
        self.header_label = tk.Label(root,
                                     text="PPE DETECTION AT CONSTRUCTION SITES:\n an application of object detection",
                                     font=("Cursive", 22, "bold"), bg='#C6C8EE', fg='#232020')
        self.header_label.pack(pady=10)

        self.video_label = tk.Label(root)
        self.video_label.pack(padx=10, pady=10)

        self.browse_button = tk.Button(root, text="Select Source", command=self.browse_video, font=("Cursive", 12),
                                       bg='#266DD3', fg='#FCFCFF', borderwidth=2, relief="solid", padx=10, pady=5, bd=0,
                                       cursor="hand2")
        self.browse_button.pack(pady=15)

        self.detect_button = tk.Button(root, text="Start Detection", command=self.start_detection, font=("Cursive", 12),
                                       bg='#6BAA75', fg='#FCFCFF', borderwidth=2, relief="solid", padx=10, pady=5, bd=0,
                                       cursor="hand2")
        self.detect_button.pack(pady=15)

        self.quit_button = tk.Button(root, text="Quit", command=root.destroy, font=("Cursive", 12), bg='#A30B37',
                                     fg='#FCFCFF', borderwidth=2, relief="solid", padx=10, pady=5, bd=0, cursor="hand2")
        self.quit_button.pack(pady=15)

    ##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def browse_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
        if file_path:
            self.video_path = file_path

    ##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def start_detection(self):
        cap = cv2.VideoCapture(self.video_path)
        while True:
            success, img = cap.read()
            if not success:
                break
            results = self.model(img, stream=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    # w, h = x2 - x1, y2 - y1

                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    current_class = self.class_names[cls]

                    if conf > 0.5:
                        if current_class == 'NO-Hardhat' or current_class == 'NO-Safety Vest' or current_class == "NO-Mask":
                            self.my_color = (0, 0, 255)
                        elif current_class == 'Hardhat' or current_class == 'Safety Vest' or current_class == "Mask":
                            self.my_color = (0, 255, 0)
                        else:
                            self.my_color = (255, 0, 0)

                        cvzone.putTextRect(img, f'{self.class_names[cls]} {conf}',
                                           (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=self.my_color,
                                           colorT=(255, 255, 255), colorR=self.my_color, offset=5)
                        cv2.rectangle(img, (x1, y1), (x2, y2), self.my_color, 3)
            ##-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)
            self.video_label.img = img
            self.video_label.config(image=img)
            self.root.update_idletasks()
            self.root.update()
        cap.release()
if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.geometry("650x500")
    root.mainloop()
