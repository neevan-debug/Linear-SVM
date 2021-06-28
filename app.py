from flask import Flask,render_template,request
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import cvxopt.solvers

app = Flask(__name__)


@app.route("/")
def decision():
    return render_template("home.html")

@app.route('/main',methods = ['POST'])
def home():
    form_data = request.form
    flag = []
    for i in form_data.values():
        flag.append(int(i))

    if flag[0]==1:
        ip = np.array([[1.682, -11.852],[0.386, 16.851],[-1.913, -11.315],[-1.754, 4.084],[-1.656, -10.834],[0.655, -8.111],
                   [-0.704, 5.832],[2.704, -10.758],[-2.656, -3.552],[0.861, 8.853],[0.975, 19.607],[3.621, -2.048],
                   [-1.195, -3.235],[1.202, 10.168],[3.193, 11.248]])
                   
        return render_template("index.html",ip = ip)
    else:
        return render_template("empty.html")

@app.route('/plot',methods = ['POST'])
def build_plot():
    form_data = request.form
    ip = []
    for i in form_data.values():
        ip.append(float(i))

    ip = np.reshape(ip,(15,2))
    
    img = io.BytesIO()
    
    plt.scatter(ip[:,0],ip[:,1],c="black")
    for i in range(15):
        j = 0
        plt.annotate(i+1, (ip[i][j], ip[i][j+1]))
    plt.savefig(img,format = 'png')
    img.seek(0)
    

    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.switch_backend('agg')

    #return '<img src="data:image/png;base64,{}">'.format(plot_url)
    return render_template("results.html",ip = ip,plot_url = plot_url)


@app.route('/result',methods = ['POST'])
def final():
    form_data = request.form
    global ip
    ip = []
    for i in form_data.values():
        ip.append(float(i))

    ip = np.reshape(ip,(15,3))
    ip_s = ip[ip[:,2].argsort()]
    X = ip_s[:,[0,1]]
    y = ip_s[:,2]

    m = X.shape[0]
    K = np.array([np.dot(X[i], X[j])
    for j in range(m)
    for i in range(m)]).reshape((m, m))

    P = cvxopt.matrix(np.outer(y, y) * K)
    q = cvxopt.matrix(-1 * np.ones(m))

    A = cvxopt.matrix(y, (1, m))
    b = cvxopt.matrix(0.0)

    G = cvxopt.matrix(np.diag(-1 * np.ones(m)))
    h = cvxopt.matrix(np.zeros(m))

    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    multipliers = np.ravel(solution['x'])
    has_positive_multiplier = multipliers > 1e-7
    sv_multipliers = multipliers[has_positive_multiplier]
    support_vectors = X[has_positive_multiplier]
    support_vectors_y = y[has_positive_multiplier]

    def compute_w(multipliers, X, y):
        return np.sum(multipliers[i] * y[i] * X[i]
        for i in range(len(y)))
    global w
    w = compute_w(sv_multipliers, support_vectors, support_vectors_y)
    print(w)

    def compute_b(w, X, y):
        return np.sum([y[i] - np.dot(w, X[i])
        for i in range(len(X))])/len(X)
    global bia
    bia = compute_b(w, support_vectors, support_vectors_y)

    img = io.BytesIO()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x = np.linspace(-4,4,100)

    plt.plot(x, (-w[0]*x - bia)/w[1], '-r')
    plt.plot(x, (-w[0]*x - bia-1)/w[1], '-.r')
    plt.plot(x, (-w[0]*x - bia+1)/w[1], '-.r')

    plt.scatter(ip[:,0],ip[:,1],c=ip[:,2])

    plt.savefig(img,format = 'png')
    img.seek(0)
    

    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.switch_backend('agg')

    
    return render_template("final.html",ip = ip,plot_url = plot_url,w = w , bia = bia)


@app.route('/calc',methods = ['POST'])
def calculate():
    form_data = request.form
    op = []
    for i in form_data.values():
        op.append(float(i))
    
    img = io.BytesIO()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x = np.linspace(-4,4,100)

    plt.plot(x, (-w[0]*x - bia)/w[1], '-r')
    plt.plot(x, (-w[0]*x - bia-1)/w[1], '-.r')
    plt.plot(x, (-w[0]*x - bia+1)/w[1], '-.r')

    plt.scatter(ip[:,0],ip[:,1],c=ip[:,2])
    plt.scatter(op[0],op[1],c="blue")
    plt.savefig(img,format = 'png')
    img.seek(0)
    
    y = np.dot(w,op) + bia
    if y >= 0:
        y = 1
    else:
        y = -1
    
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.switch_backend('agg')

    #return '<img src="data:image/png;base64,{}">'.format(plot_url)
    return render_template("cal.html",op = op,plot_url = plot_url,y = y)




if __name__ == "__main__":
    app.run(debug=True,threaded = True)