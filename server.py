from flask import Flask, Response,request,render_template
from synth import Synth
app = Flask(__name__,template_folder='', static_folder='',static_url_path='')

@app.route("/")
def main():
	return render_template('synth2.html')
		
@app.route("/synthesize")
def syn():
		text = request.args.get('text')
		wave = s.synth(text)
		return Response(wave, mimetype='audio/wav')
s = Synth()		
if __name__ == "__main__":
	app.run(debug=True)