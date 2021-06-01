from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

def define_model():
    a=var.get()
    if(a==1):
        messagebox.showinfo(title="Mean Absolute Error",message="Mean Absolute Error: "+str("{:.2f}".format(svm_err)))
    elif(a==2):
        messagebox.showinfo(title="Mean Absolute Error", message="Mean Absolute Error: " + str("{:.2f}".format(rfc_err)))
    else:
        messagebox.showinfo(title="Mean Absolute Error", message="Mean Absolute Error: " + str("{:.2f}".format(knn_err)))
    pass

def calculate_accuracy(df):
    x_data = df[['SepalLengthCm','SepalWidthCM','PetalLengthCm','PetalWidthCm']]
    y_data = df['irisClass']
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, shuffle=True)
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    svm_acc=metrics.accuracy_score(y_test, y_pred)
    global svm_err
    svm_err=mean_absolute_error(y_test, y_pred)
    print("Accuracy:", svm_acc)

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    rfc_acc = metrics.accuracy_score(y_test, y_pred)
    global rfc_err
    rfc_err=mean_absolute_error(y_test, y_pred)
    print("Accuracy:", rfc_acc)

    knn=KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    knn_acc = metrics.accuracy_score(y_test, y_pred)
    global knn_err
    knn_err=mean_absolute_error(y_test, y_pred)
    print("Accuracy:", knn_acc)

    global var
    var = IntVar()
    r1=Radiobutton(f2,text="SVM Accuracy "+str("{:.2f}".format(svm_acc)),variable=var,value=1)
    r1.grid(row="3",column='1',sticky=W)
    r1.select()
    r2 = Radiobutton(f2, text="RFC Accuracy " + str("{:.2f}".format(rfc_acc)), variable=var,value=2).grid(row="4", column='1',sticky=W)
    r3 = Radiobutton(f2, text="KNN Accuracy " + str("{:.2f}".format(knn_acc)), variable=var,value=3).grid(row="5", column='1',sticky=W)
    Label(f2, text='Step 3:', height='1', padx=5, pady=2, bd=3).grid(row=6, column=0)
    Label(f2, text='Define the model', height='1', padx=5, pady=2,bd=3).grid(row=6, column=1, sticky=W)
    Button(f2, text="Calculate accuracy", height='1', width="18", padx=5, pady=2, bd=3, font=("Helvetica", 8),command=define_model).grid(row=6, column=2)

    root.geometry("705x250")

def load_csv_dataset():
    csv_file=askopenfilename(title='Load csv Dataset',filetypes=[("csv", ".csv")])
    e.delete(0, END)
    e.insert(0, csv_file)
    df=pd.read_csv(csv_file)
    Label(f2,text='').grid(row=1,column=0)
    root.geometry("705x140")
    Label(f2, text='Step 2:', height='1', padx=5, pady=2, bd=3).grid(row=2,column=0)
    Label(f2, text='Apply machine learning algorithms & select the best Accuracy:', height='1', padx=5, pady=2, bd=3).grid(row=2,column=1,sticky=W)
    Button(f2,text="Calculate accuracy", height='1', width="18", padx=5, pady=2, bd=3, font=("Helvetica", 8),command=lambda: calculate_accuracy(df)).grid(row=2,column=2)



root = Tk()
root.geometry("705x100")
root.title(".:: Guess Iris Plant Type ::.")
f1=Frame(root)
f1.pack()
Label(f1, text = '.:: Guess Iris Plant Type ::.',bg='skyblue',fg='white',padx=2,pady=2,bd=3,font=("Helvetica", 25)).pack(side=TOP)
f2=Frame(root)
f2.pack()
Label(f2, text = 'Step 1:',height='1',padx=5,pady=2,bd=3).grid(row=0,column=0)
e= Entry(f2,width='86',bd=2)
e.grid(row=0,column=1)
Button(f2,text="Load CSV Dataset",height='1', width="18",padx=5,pady=2,bd=3, font=("Helvetica", 8), command=load_csv_dataset).grid(row=0,column=2)
root.mainloop()
