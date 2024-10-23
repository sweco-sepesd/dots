from flask import Flask, render_template, request
import cv2
import threading
import detectandshow4  # Make sure this script has a main function to capture an image

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('oredev.html')

@app.route('/submit', methods=['POST'])
def run_script():
    # Logic to run your script here
    name = request.form['name']
    email = request.form['email']
    #processedtext = f"Name: {name}, Email: {email}"
    # Create a thread to run the capture script
    capture_thread = threading.Thread(target=detectandshow4.main)
    capture_thread.start()
    
    return "Image capture initiated!"  # You can modify this to redirect or render a success template
    
    
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4996)


