#request-for accessing file which was uploaded by the user on our application.
import os
import tensorflow as tf
global graph
graph = tf.compat.v1.get_default_graph()
import numpy as np  # used for numerical analysis
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

from flask import Flask, render_template, request, url_for
from keras.preprocessing import image
from keras.models import load_model  # to load our trained model

app = Flask(__name__,template_folder="templates") # initializing a flask app
# Loading the model
model=load_model('Fruites.h5')
print("Loaded model from disk")


@app.route('/')# route to display the home page
def home():
    return render_template('Index.html')#rendering the home page

@app.route('/analyser',methods=['GET','POST'])# routes to the index html
def image1():
    return render_template("Analyzer.html") 



@app.route('/ai',methods=['GET', 'POST'])# route to show the predictions in a web UI
def launch():
    if request.method=='POST':
        f=request.files['image'] 
        print('current path')#requesting the file
        basepath=os.path.dirname('__file__')
        print('current path',basepath)#storing the file directory
        filepath=os.path.join(basepath,"uploads",f.filename)
        print('upload folder is',filepath)#storing the file in uploads folder
        f.save(filepath)#saving the file
        
        img=image.load_img(filepath,target_size=(64,64)) #load and reshaping the image
        x=image.img_to_array(img)#converting image to an array
        x=np.expand_dims(x,axis=0)#changing the dimensions of the image

       
        with graph.as_default():
            pred=model.predict_classes(x)
            print("prediction",pred)#printing the prediction
        index=['APPLES','BANANA','ORANGE','PINEAPPLE','WATERMELON']
        text = 'the predicted animal is: ' + str(index[pred[0]])
        
        return text
    else:
        return render_template('Result.html')

if __name__ == "__main__":
   # running the app
    app.run(debug=False)
