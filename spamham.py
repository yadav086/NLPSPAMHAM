from flask import Flask, request, render_template,jsonify
import pickle
app= Flask(__name__)
model = pickle.load(open('spamham_ps.pkl','rb'))
tif = pickle.load(open('tranform.pkl','rb'))

@app.route('/')
def get_text():
	#return ''' This is home page of NLP'''
	return render_template('/NLP/NLP_home.html')

@app.route('/predict',methods=["POST"])
def get_predict():

	data = request.form['nlp']
	message = [data]
	vec = tif.transform(message).toarray()
	mypredict = model.predict(vec)
	mypredict= mypredict[0]

	if mypredict ==0:
		mypredict='HAM'
	else:
		mypredict='SPAM'


	return render_template('/NLP/nlp_result.html',mypredict=mypredict)

if __name__== '__main__':
	app.run(debug= True)