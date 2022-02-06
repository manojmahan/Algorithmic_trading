from flask import Flask ,render_template ,request

import model

app = Flask(__name__)

@app.route("/")

def hello():
    if request.method=="POST":
        name = request.form["company_name"]
        output = model.My_model(name)
        print(output)
    
    return render_template("index.html")


#@app.route("/",methods=['POST'])
#def submit():
    if request.method=="POST":
        name = request.form["company name"]
        model.My_model(name)
    return render_template("sub.html",n=name)

if __name__ == '__main__':
    app.run(debug=True)