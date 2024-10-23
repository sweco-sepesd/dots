from flask import Flask, render_template, request
import cv2
import threading
import detectandshow4  # Make sure this script has a main function to capture an image
import os.path
import post_results2
from urllib.error import HTTPError
from post_results2 import upload_resource
from post_results2 import submit_job

app = Flask(__name__)

# Global variable to store the image UUID
image_uuid = None
lock = threading.Lock()

def run_capture():
    global image_uuid
    image_uuid = detectandshow4.main()  # Call the main function and get the UUID


@app.route('/')
def home():
    return render_template('oredev.html')

@app.route('/submit', methods=['POST'])
def run_script():
    # Logic to run your script here
    global image_uuid  # Use the global variable
    name = request.form['name']
    email = request.form['email']
    with lock:
        thread = threading.Thread(target=run_capture)
        thread.start()
        thread.join()
        print (image_uuid)
        

        img_name = "{}.png".format(image_uuid)
        print(img_name)
 
        fp = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','my_flask_app', img_name))
        with open(fp, 'rb') as fin:
            data = fin.read()
            print ("data read")
            try:
                path = 'oredev'
                file_info = upload_resource(data, img_name, 'FME_SHAREDRESOURCE_DATA', path, overwrite=True)
                print(file_info)
                # NOTE: Dataset parameters for file based reader formats are often expected to be an array (that's how multiple files are handled)
                job_info = submit_job('CONNECT_THE_DOTS', 'evaluate_contribution.fmw', {'ANVANDARE': name, 'UUID': str(image_uuid), 'EPOST': email})
                print(job_info)
            except HTTPError as e:
                print(e.msg, e.name, e.reason, e.status)
   
        return "Image captured and processed, you will recieve your results by email, you can also see them on the screen in the display"  # You can modify this to redirect or render a success template

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4996)


