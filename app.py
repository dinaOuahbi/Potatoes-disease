from flask import Flask, render_template, url_for, request, redirect, flash
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

UPLOAD_FOLDER = 'C:/Users/Admin/tf_application/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

MODEL = tf.keras.models.load_model('saved_models/1')
CLASS_NAMES = ['Early blight', 'Late blight', 'healthy']

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(data))
    return image

@app.route('/result', methods=['GET', 'POST'])
def result():
    predicted_class = request.args.get('predicted_class')
    confidence = request.args.get('confidence')
    return render_template('result.html', predicted_class=predicted_class, confidence=confidence)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # DL ------------------------------------
            image = read_file_as_image(f"uploads/{filename}")
            img_batch = (np.expand_dims(image, 0)) # add dim in row
            predictions = MODEL.predict(img_batch)
            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            confidence = np.max(predictions[0])
            #return f"{predicted_class} {confidence}"
            # end ----------------------------------------
            return redirect(url_for('result', predicted_class=predicted_class, confidence=confidence))
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
