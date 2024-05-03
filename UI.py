import tkinter as tk
import cv2
from PIL import Image, ImageTk

def play_video(window, video_label):
    global cap  # Declare cap as global to access it inside the function
    cap = cv2.VideoCapture('video.mp4')  # Replace 'video.mp4' with the actual path to your video file

    def update_frame():
        ret, frame = cap.read()
        if ret:
            # Convert the frame from OpenCV BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize the frame to 700x700
            frame = cv2.resize(frame, (700, 700))
            # Convert the frame to PIL Image
            image = Image.fromarray(frame)
            # Convert the PIL Image to Tkinter PhotoImage
            photo = ImageTk.PhotoImage(image=image)

            # Update the label with the new PhotoImage
            video_label.config(image=photo)
            video_label.image = photo

            # Schedule the next update
            window.after(25, update_frame)
        else:
            # Release the video capture and destroy the window
            cap.release()
            cv2.destroyAllWindows()
            video_label.pack_forget()  # Hide the video label

    # Start the update loop
    update_frame()

def start_clicked(window, button_start):
    button_start.pack_forget()  # Hide the start button
    # Create a label to display the video
    video_label = tk.Label(window)
    video_label.pack(anchor='center')  # Pack the label into the center of the window
    play_video(window, video_label)
    # Create a stop button 
    button_stop = tk.Button(window, text="Stop", bg='white', fg='black', font=('Helvetica', 20), bd=2, highlightthickness=2, command=lambda: stop_clicked(window, button_start, button_stop))
    button_stop.place(relx=0.5, rely=0.6, anchor='center')  
    button_stop.config(width=10, height=3)
    #button_stop.pack_forget()  # Hide the stop button initially
    button_stop.pack()  # Show the stop button

def stop_clicked(window, button_start, button_stop):
    button_stop.pack_forget()  # Hide the stop button
    # Stop the video playback and return the start button
    global cap  # Access the global cap variable
    if cap.isOpened():
        cap.release()
    window.video_label.pack_forget()  # Hide the video label
    button_start.pack()  # Show the start button

def quit_clicked(window):
    window.destroy()  # Close the Tkinter window, exiting the application

# Create the main window
window = tk.Tk()
window.title("UI Example")
window.geometry('900x900')
window.configure(bg='black')

# Create a start button
button_start = tk.Button(window, text="Start", bg='white', fg='black', font=('Helvetica', 20), bd=2, highlightthickness=2, command=lambda: start_clicked(window, button_start))
button_start.place(relx=0.5, rely=0.5, anchor='center')  
button_start.config(width=10, height=3)

window.mainloop()
