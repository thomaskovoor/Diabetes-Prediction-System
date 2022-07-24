from flask import Flask,render_template,request,redirect,url_for
import numpy as np
import pickle as pk

app = Flask(__name__)

#load model using 'pickle'
def model_call():
    with open("main-model","rb") as file:
        model = pk.load(file)
        return model

#replace zero values with their respective means
def replace_zero(user_array):
    data = np.loadtxt("mean_values.txt")
    for i in range(1,7):
        if(user_array[0][i] == 0):
            user_array[0][i] = data[i]
    return user_array

@app.route('/positive',methods=["GET"] )
def positive():
    return render_template("positive_proto.html")

@app.route('/negative',methods=["GET"] )
def negative():
    return render_template("negative_proto.html")

@app.route('/',methods=["POST","GET"] )
def home():
    if request.method == "POST":
        #each element from the form
        ele = [float(x) for x in request.form.values()]
        user_array = np.array(ele)
        user_array = np.reshape(user_array, (-1,8))
        print(user_array)

        #uncomment later
        # user_array = replace_zero(user_array)
        # print(user_array)
        
        
        model = model_call()
        pred = model.predict(user_array)
        print(pred)
        
        if(pred == 1):
            return redirect(url_for("positive"))
        else:
            return redirect(url_for("negative"))
    else:    
        return render_template("index.html")

if __name__ == "__main__" :
    app.run(debug=True)
