from flask import Flask, flash,request,redirect,url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename
import flask_roman_numerals
import os
import generate_roman_numerals

UPLOAD_FOLDER = './Audio'
#UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/Audio/'
CUSTOM_STATIC = './Audio'
ALLOWED_EXTENSIONS = set(['wav'])

app = Flask(__name__, template_folder='template')

app.config['CUSTOM_STATIC_PATH'] = CUSTOM_STATIC
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    #print(filename.rsplit('.', 1)[1].lower())
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/HowItWorks")
#@app.route("/HowItWorks.html")
def how():
    return render_template('HowItWorks.html')

@app.route("/examples", methods=['GET','POST'])
def examples():
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
            #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            res = generate_roman_numerals.generate(filename)
            key = res[0]
            roman_nums = res[1]
            print(filename)
            return render_template('examples.html',
                                    key=key, chords=roman_nums)
    return render_template('examples.html')

@app.route('/examples/<path:filename>')
def custom_static(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/about")
def about():
    return render_template('about.html')


if __name__ == "__main__":
    app.run()
