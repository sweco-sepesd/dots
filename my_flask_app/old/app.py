from flask import Flask, render_template, request
import subprocess
import cv2

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('oredev.html')  # Render your HTML page

@app.route('/submit', methods=['POST'])
def run_script():
    # Logic to run your script here
    name = request.form['name']
    email = request.form['email']
    #processedtext = f"Name: {name}, Email: {email}"
    
    # Call OpenCV script
    # Command to run the other Python script
    command = ["python", "detectandshow2.py"]
    result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while True:
        k = cv2.waitKey(10) & 0xFF
        if k == ord('p'):
            # Send 'p' to the second script
            process.stdin.write('p\n')
            process.stdin.flush()
            break
    #result.wait()
    
        
    output, error = result.communicate()
    output = output.decode('utf-8')
    error = error.decode('utf-8')
    # Get the output (uuid)from detectandshow
    if error:
        print("Error:", error)
        
    print(output)
    return output

    #cmd ='"C:/Program Files/FME/fme.exe" "C:/Users/SE1E2U/Desktop/Skane_server.fmw" --ANVANDARE "'+name+'" --EPOST "'+email+'"'
    #process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    #process.wait()




    #return redirect(url_for('home'))  # Redirect after processing
    return output
    #return processedtext  # For demonstration; you may want to render a different template

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4996)
