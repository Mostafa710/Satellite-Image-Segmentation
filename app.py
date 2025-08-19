from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from utils import load_model, predict_mask
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static"
ALLOWED_EXTENSIONS = {'tif', 'tiff'}

model = load_model("model/UNet_with_MobileNetV2_backbone.pth")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clear_static_folder():
    keep_files = {"style.css", "script.js"}
    for filename in os.listdir(app.config["UPLOAD_FOLDER"]):
        if filename in keep_files:
            continue  # skip CSS and JS files
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


@app.route("/", methods=["GET", "POST"])
def index():
    clear_static_folder()  # delete all previous session files

    if request.method == "POST":
        file = request.files["image"]
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            output_path = os.path.join(app.config["UPLOAD_FOLDER"], "output.png")
            input_rgb_path = os.path.join(app.config["UPLOAD_FOLDER"], "input_rgb.png")

            predict_mask(model, filepath, output_path, input_rgb_path=input_rgb_path)

            return render_template("index.html",
                                   input_image="input_rgb.png",
                                   result_image="output.png")
        else:
            return "Invalid file format. Please upload a .tif image."

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
