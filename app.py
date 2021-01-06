from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = './UPLOAD_FOLDER/'

@app.route('/')
def hello():
    return 'Welcome to intelligent gate app backend !'


@app.route('/visitor/emotion/check', methods=['POST', 'GET'])
def get_image():
    filename = UPLOAD_FOLDER + str(np.random.randint(0, 5000)) + '.png'
    print('Image is incoming')
    photo = request.files['photo']
    photo.save(filename)
    print('Image Saved..')
    return 'image saved'


if __name__ == '__main__':
    app.run(debug=any)
