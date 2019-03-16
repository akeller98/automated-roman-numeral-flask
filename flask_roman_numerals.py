from flask import Flask, flash,request,redirect,url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename
import flask_roman_numerals
import os
import generate_roman_numerals

UPLOAD_FOLDER = './Audio'
ALLOWED_EXTENSIONS = set(['wav'])

app = Flask(__name__, template_folder='template')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
file_to = ''

def allowed_file(filename):
    print(filename.rsplit('.', 1)[1].lower())
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
@app.route("/index.html")
def home():
    return render_template('index.html')

@app.route("/HowItWorks.html")
def how():
    return render_template('HowItWorks.html')

@app.route("/examples.html", methods=['GET','POST'])
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
            file.filename = 'upload.wav'
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            res = generate_roman_numerals.generate(filename)
            key = res[0]
            roman_nums = res[1]
            return render_template('examples.html',
                                    name=key, chords=roman_nums)
    return render_template('examples.html')

@app.route("/about.html")
def about():
    return render_template('about.html')

@app.route("/upload", methods=['GET','POST'])
def upload_file():
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
            file.filename = 'upload.wav'
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('generateRoman',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Automated Roman Numeral Analysis</title>
    <h1>Automated Roman Numeral Analysis</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/roman/<filename>')
def generateRoman(filename):
    #key = os.system('python generate_roman_numerals.py ' + filename)
    res = generate_roman_numerals.generate(filename)
    key = res[0]
    roman_nums = res[1]
    return render_template('results.html', name=key, chords=roman_nums)

if __name__ == "__main__":
    app.run()
