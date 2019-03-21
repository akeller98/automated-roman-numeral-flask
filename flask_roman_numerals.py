from flask import Flask, flash,request,redirect,url_for, send_from_directory, render_template, jsonify
from werkzeug.utils import secure_filename
import flask_roman_numerals
import os
import generate_roman_numerals

UPLOAD_FOLDER = './Audio'
ALLOWED_EXTENSIONS = set(['wav'])

app = Flask(__name__, template_folder='template')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/HowItWorks")
def how():
    return render_template('HowItWorks.html')

@app.route("/try_me", methods=['GET','POST'])
def try_me():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            file.filename = 'upload_new.wav'
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            res = generate_roman_numerals.generate(filename)
            key = res[0]
            roman_nums = res[1]
            print(filename)
            return jsonify({"key": key, "roman_nums": roman_nums})

    return render_template('try_me.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/testing")
def testing():
    return render_template('testing.html')

@app.route("/future")
def future():
    return render_template('future.html')

if __name__ == "__main__":
    app.run()
