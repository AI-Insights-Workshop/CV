
loaded_model = load_model('cifar10_vgg19_model.h5')

# # 6. Hosting the Model
# # Simple Flask app for hosting the model
# from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    img = request.files['image'].read()
    img = cv2.imdecode(np.fromstring(img, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (32, 32))
    img = np.expand_dims(img, axis=0)
    prediction = loaded_model.predict(img)
    class_idx = np.argmax(prediction, axis=1)[0]
    return jsonify({'class': int(class_idx)})

if __name__ == '__main__':
    app.run(debug=True)