# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

import numpy as np
import dietApi
import utils.bmi as bmi
# Flask constructor takes the name of 
# current module (__name__) as argument.
app = Flask(__name__)
allowed_origins = [""]

CORS(app, supports_credentials=True, origins=allowed_origins)

# The route() function of the Flask class is a decorator, 
# which tells the application which URL should call 
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
	return 'Hello World'

@app.route('/recommend_diet', methods=['POST'])
def recommend_diet():
    data = request.json
    recommendation = dietApi.recommend_dish(data)
    print(recommendation)
    # Convert the NumPy array to a list for JSON serialization
    recommendation_list = recommendation.tolist() if isinstance(recommendation, np.ndarray) else recommendation

    return jsonify(recommendation_list)
@app.route('/diet',methods=['GET'])
def diet():
	data=request.json
	user_input=dietApi.show_entry_fields(data)
	print(user_input)
	return jsonify(user_input)
@app.route('/bmi',methods=['GET'])
def bmiCalculation():
	data=request.json
	_,_,_,output,_=bmi.bmiCalculation(data)
	return jsonify(output)
@cross_origin(origins='*')
@app.route('/healthy-diet', methods=['POST'])
def healthyDiet():
    user_data = request.json
    response = dietApi.Healthy(user_data)
    return jsonify(response)


# main driver function
if __name__ == '__main__':

	# run() method of Flask class runs the application 
	# on the local development server.
	app.run()
