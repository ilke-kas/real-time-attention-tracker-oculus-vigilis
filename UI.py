import tkinter as tk
import cv2
from PIL import Image, ImageTk
import math
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import sys
import pytorch_lightning as pl
from torchmetrics import Accuracy
import torch.optim as optim
import matplotlib.pyplot as plt
import pdfkit
import logging
import torch
import threading
sys.path.append('./DatasetPreparation/')
from extractor import FeatureExtractor
from torch.utils.data import Dataset, DataLoader
stopped=False

class AttentionDataset(Dataset):
  def __init__(self,sequences):
    self.sequences = sequences
  def __len__(self):
    return len(self.sequences)
  def __getitem__(self,idx):
    sequence = self.sequences[idx]
    return dict(
        sequence=torch.Tensor(sequence.to_numpy())
    )

class AttentionModel(nn.Module):
  def __init__(self, n_features, n_classes, n_hidden=256, n_layers = 3):
    super().__init__()

    self.lstm = nn.LSTM(
        input_size = n_features,
        hidden_size = n_hidden,
        num_layers=n_layers,
        batch_first=True,
        dropout=0.75
    )

    self.classifier = nn.Linear(n_hidden, n_classes)

  def forward(self,x):
    self.lstm.flatten_parameters()
    _, (hidden,_) = self.lstm(x)

    out = hidden[-1]
    return self.classifier(out)

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

accuracy = Accuracy(task="multiclass", num_classes=4).to(device)
class AttentionPredictor(pl.LightningModule):
  def __init__(self, n_features: int, n_classes: int):
    super().__init__()
    self.model = AttentionModel(n_features, n_classes)
    self.criterion = nn.CrossEntropyLoss()

  def forward(self,x,labels=None):
    output= self.model(x)
    loss = 0
    if labels is not None:
      loss= self.criterion(output,labels)
    return loss, output
  def training_step(self,batch,batch_idx):
    sequences= batch["sequence"]
    labels = batch["label"]
    loss, outputs = self(sequences,labels)
    predictions = torch.argmax(outputs, dim=1)
    step_accuracy = accuracy(predictions,labels)

    self.log("train_loss",loss, prog_bar=True, logger=True)
    self.log("train_accuracy",step_accuracy, prog_bar=True, logger=True)
    return {"loss":loss,"accuracy":step_accuracy}

  def validation_step(self,batch,batch_idx):
    sequences= batch["sequence"]
    labels = batch["label"]
    loss, outputs = self(sequences,labels)
    predictions = torch.argmax(outputs, dim=1)
    step_accuracy = accuracy(predictions,labels)

    self.log("val_loss",loss, prog_bar=True, logger=True)
    self.log("val_accuracy",step_accuracy, prog_bar=True, logger=True)
    return {"loss":loss,"accuracy":step_accuracy}

  def test_step(self,batch,batch_idx):
    sequences= batch["sequence"]
    labels = batch["label"]
    loss, outputs = self(sequences,labels)
    predictions = torch.argmax(outputs, dim=1)
    step_accuracy = accuracy(predictions,labels)

    self.log("test_loss",loss, prog_bar=True, logger=True)
    self.log("test_accuracy",step_accuracy, prog_bar=True, logger=True)
    return {"loss":loss,"accuracy":step_accuracy}

  def configure_optimizers(self):
    return optim.Adam(self.parameters(),lr= 0.0001)


model = AttentionPredictor(
    n_features=4,
    n_classes=3
    )
all_predictions = []
model_path = "./86_acc_trained_model.pth"  # Specify the path to your .pth file
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
class VideoThread(threading.Thread):
    def __init__(self, window, video_label,average_label):
        super().__init__()
        self.window = window
        self.video_label = video_label
        self.average_label = average_label 
        self._stop_event = threading.Event()

    def run(self):
            feature_extractor = FeatureExtractor()
            self.cap = cv2.VideoCapture(0)  
            frames = [] 
            chunk_index = 0
            # Read until video is completed
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (700, 700))
                    image = Image.fromarray(frame)
                    photo = ImageTk.PhotoImage(image=image)

                    self.video_label.config(image=photo)
                    self.video_label.image = photo

                    #send  it to the feature extractor
                    frames.append(frame)
                    if len(frames) >= 100:
                        while chunk_index < math.floor(len(frames)/100):
                            chunk_frames = frames[chunk_index*100:(chunk_index+1)*100]
                            features = feature_extractor.extract_features(chunk_frames, '1_1', chunk_index)
                            df = pd.DataFrame(features)
                            FEATURE_COLUMNS = ['ear', 'lip_distance', 'face_pose', 'iris_pose']
                            sequences = []
                            for index, group in  df.groupby(['video_name','chunk_index']):
                                sequence_features = group[FEATURE_COLUMNS]
                                sequences.append(sequence_features)
                            test_dataset = AttentionDataset(sequences)
                            for item in tqdm(test_dataset):
                                sequence = item["sequence"]
                                _,output = model(sequence.unsqueeze(dim=0).to(device))

                                prediction = torch.argmax(output,dim=1)
                                print("Current attention prediction is " + str(prediction))
                                all_predictions.append(prediction.item()+1)
                            # Take the last 3 elements if the array has more than 3 elements, otherwise take all elements
                            last_three = all_predictions[-3:] if len(all_predictions) >= 3 else all_predictions

                            if last_three:
                                average = sum(last_three) / len(last_three)
                            else:
                                average = 0  # Or any other default value you prefer if the array is empty

                            print("Average of last three elements:", average)
                            # Update the average label text with the calculated average score
                            self.average_label.config(text="Average Score: {:.2f} /3.00".format(average))

                            chunk_index=chunk_index+1

                else: 
                    break
            self.cap.release()
    def stop(self):
        stopped =True
        if self.cap.isOpened():
            self.cap.release()
        self._stop_event.set()
         # Set the event to signal thread termination

def start_clicked(window, button_start, video_thread,average_label):
    
    button_start.pack_forget()
    video_label = tk.Label(window)
    video_label.pack(anchor='center')

    video_thread = VideoThread(window, video_label, average_label)
    video_thread.start()
    # Bind the closing event to the on_closing function
    window.protocol("WM_DELETE_WINDOW", lambda: on_closing(window,video_thread))
    button_stop = tk.Button(window, text="Stop", bg='white', fg='black', font=('Helvetica', 20), bd=2, highlightthickness=2, command=lambda: stop_clicked(window, button_start, button_stop, video_thread))
    button_stop.place(relx=0.5, rely=0.6, anchor='center')  
    button_stop.config(width=10, height=3)
    button_stop.pack()
# After the while loop where you calculate the average attention level
# Define a function to plot time vs attention level
def plot_attention_over_time():
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(all_predictions)), all_predictions, marker='o', linestyle='-')
    plt.title('Time vs Attention Level')
    plt.xlabel('Time')
    plt.ylabel('Attention Level')
    plt.grid(True)
    plt.tight_layout()

def stop_clicked(window, button_start, button_stop, video_thread):
    # After calculating the average attention level
    # Call the function to plot time vs attention level
    plot_attention_over_time()
    # Save the plot as a PDF
    plt.savefig('attention_over_time.pdf')
    # Close the plot
    plt.close()
    video_thread.join(5)
    video_thread.stop()
    button_stop.pack_forget()
  
    #window.destroy()
    #raise Exception("Program Stopped")

def on_closing(window,video_thread): 
    video_thread.stop()
    window.destroy()

def main_window():
    window = tk.Tk()
    window.title("UI Example")
    window.geometry('900x900')
    window.configure(bg='black')

    button_start = tk.Button(window, text="Start", bg='white', fg='black', font=('Helvetica', 20), bd=2, highlightthickness=2, command=lambda: start_clicked(window, button_start, None, average_label))
    button_start.place(relx=0.5, rely=0.5, anchor='center')  
    button_start.config(width=10, height=3)

    average_label = tk.Label(window, text="Average Score: 0.00/3.00", font=('Helvetica', 14), bg='black', fg='white')
    average_label.pack()

    if not stopped:
        window.mainloop()
    else:
        print("here")
        window.quit()


if __name__ == "__main__":
    main_window()





