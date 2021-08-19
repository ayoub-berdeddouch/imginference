from flask import Flask, render_template, request
from flask import json
import logging

# uncomment this after 

#import tensorflow as tf
#from keras.models import load_model
#from keras.preprocessing import image

app = Flask(__name__,template_folder='templates')


## Uncomment this after.

# Prediction or Classification
# def predict_label(img_path):
# 	i = image.load_img(img_path, target_size=(100,100))
# 	i = image.img_to_array(i)/255.0
# 	i = i.reshape(1, 100,100,1)
# 	p = model.predict_classes(i)
# 	return dic[p[0]]

# routes
@app.route("/",methods=['GET','POST'])
def main():

	# log line
	app.logger.info('Welcome Request successfull')

	return render_template('index.html')



@app.route("/classification", methods=['GET','POST'])
def heart_risk():
	if request.method == 'GET':
		
		return render_template('classify.html')
	
	if request.method == 'POST':

		img = request.files['my_image']
		img_path = './static/images/' + img.filename
		img.save(img_path)

		#p = predict_label(img_path)
		p = img.filename


	return render_template("classify.html", prediction = p, img_path = img_path)




if __name__ == "__main__":

	## stream logs to app.log file
	logging.basicConfig(filename='app.log',level=logging.DEBUG)
	
	app.run(debug=True)