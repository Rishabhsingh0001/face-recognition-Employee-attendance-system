
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import datetime
import cv2
import time , os , csv
from PIL import Image, ImageTk

ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
day,month,year=date.split("-")
mont={'01':'January','02':'February','03':'March','04':'April','05':'May','06':'June','07':'July','08':'August','09':'September','10':'October','11':'November','12':'December'}

global _capture_frame , _profile_face , _mark_my_attendance_profile , _treeview
_capture_frame = None
_profile_face = None
_mark_my_attendance_profile = None

def tick(c):
    # This Updates the Clock every 200 ms.
    time_string = time.strftime('%H:%M:%S')
    c.config(text=time_string)
    c.after(200,tick,c)

def clear(label_name):
    label_name.delete(0, 'end')

def reset(t1,t2):
    t1.delete(0, 'end')
    t2.delete(0, 'end')

def update(msgbox):
    msgbox.configure( text='Total Registrations till now  : '+str(get_employee_details()) )

def update_treeview():
    global _treeview
    ## fill all the details
    if os.path.isfile("Attendance\\Attendance.csv"):
        _file = open( "Attendance\\Attendance.csv" , newline='' )
        reader = csv.reader(_file , delimiter=',')
        _idx = 1
        for k in _treeview.get_children():
            _treeview.delete(k)
        for row in reader:
            _treeview.insert('', 0, text=str(_idx), values=( row[0] , row[1] , row[2] , row[3] ))
            _idx += 1
        _file.close()

def show_frames(cap , detector , label):
    global _capture_frame
    _frame = cap.read()[1]
    gray = cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        cv2.rectangle(_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(_frame, "Register this face ? ", ( x , y+w+20 ), cv2.FONT_HERSHEY_SIMPLEX , 0.7, (0,0,255) )

    cv2image = cv2.cvtColor(_frame,cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        _capture_frame = img.crop( (x, y, x+w, y+h) )
    else:
        _capture_frame = None
    img = img.resize((690,480))

    imgtk = ImageTk.PhotoImage(image = img)

    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.after(20, lambda : show_frames(cap , detector,label))

def capture_image(cap , new_window , _info):
    global _capture_frame , _profile_face
    if _capture_frame != None:
        # _capture_frame.save("EmployeeDetails\\images\\" + str(_info[0]) + "_"+ str(_info[1].split(" ")[0]) + "_image.jpg" )
        _profile_face = _capture_frame
        print("Image Captured...")
        cap.release()
        new_window.destroy()
        tk.messagebox.showinfo( "Status" , "Image Captures Success")
    else:
        print("Failed")
        tk.messagebox.showinfo( "Status" , "No Image Detected",parent=new_window)

def new_window_for_capture(window , _info):
    new_window = tk.Toplevel(window)
    new_window.geometry("700x550")
    new_window.resizable(False,False)
    new_window.title("Capture Image")
    new_window.configure(background="white")

    label = tk.Label(new_window)
    label.grid(row=0, column=0)

    cap = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier("files\haarcascade_frontalface_default.xml")

    takeImg = tk.Button(new_window, text="Capture" , command= lambda : capture_image( cap , new_window , _info) ,fg="black"  ,bg="#F5FD03"  ,width=34  ,height=1, activebackground = "white" ,font=('times', 15, ' bold '))
    takeImg.place(x=155, y=485)

    show_frames(cap, detector, label )

def train_images():
    _images_list = os.listdir("EmployeeDetails\\images\\")
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    detector = cv2.CascadeClassifier( "files\\haarcascade_frontalface_default.xml" )
    faces, ID = getImagesAndLabels()
    try:
        recognizer.train(faces, np.array(ID))
    except Exception as e:
        print(e)
        print("Unknown Error while creating training model")
        # tk.messagebox.showinfo( "Error" , "Unknown Error.")
        return
    recognizer.save("EmployeeDetails\\TrainedData\\trainedData.yml")

def getImagesAndLabels():
    path = "EmployeeDetails\\images"
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        ID = int(os.path.split(imagePath)[-1].split("_")[0])
        faces.append(imageNp)
        Ids.append(ID)
    return faces, Ids

def take_image(window,_info):
    if (len(_info[0]) <= 0) or (len(_info[1]) <= 0 ) :
        tk.messagebox.showinfo( "Error" , "ID & NAME must entered first")
    else:
        new_window_for_capture(window,_info)

def save_profile(_info):
    if (len(_info[0].get()) <= 0) or (len(_info[1].get()) <= 0 ) :
        tk.messagebox.showinfo( "Error" , "Empty field...")
    elif _profile_face == None:
        tk.messagebox.showinfo( "Error" , "No Image is Captured yet")
    else:
        _profile_face.save("EmployeeDetails\\images\\" + str(_info[0].get()) + "_"+ str(_info[1].get().split(" ")[0]) + "_image.jpg" )
        _file = open( "EmployeeDetails\EmployeeDetails.csv" , "a" , newline='\n' )
        writer = csv.writer(_file , delimiter=',')
        # writer.writerow([ ID , NAME , DATE , TIME ])
        _info_write = [ str(_info[0].get()) , str(_info[1].get()) , str(day)+"-"+str(mont[month])+"-"+str(year) , str( time.strftime('%H:%M:%S') ) ]
        writer.writerow(_info_write)
        _file.close()
        train_images()
        reset(_info[0],_info[1])
        tk.messagebox.showinfo( "Status" , "Profile Saved Successfully !!!")

def show_frames_for_attendance(cap , detector , recognizer , all_profile_data , label):
    global _capture_frame , _mark_my_attendance_profile
    _name = "Unknown"
    _frame = cap.read()[1]
    gray = cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        cv2.rectangle(_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        s , c = recognizer.predict(gray[y:y + h, x:x + w])
        if c < 50:
            try:
                _name = all_profile_data[str(s)][1]#str(s)
                _mark_my_attendance_profile = all_profile_data[str(s)]
            except KeyError:
                _name = "Unknown"
        cv2.putText(_frame, _name , ( x , y+w+20 ), cv2.FONT_HERSHEY_SIMPLEX , 0.7, (255,0,0) )

    cv2image = cv2.cvtColor(_frame,cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        _capture_frame = img.crop( (x, y, x+w, y+h) )
    else:
        _capture_frame = None
    img = img.resize((690,480))

    imgtk = ImageTk.PhotoImage(image = img)

    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.after(20, lambda : show_frames_for_attendance(cap , detector , recognizer , all_profile_data ,label))

def mark_my_attendance( cap , new_window):
    global _mark_my_attendance_profile
    if _mark_my_attendance_profile != None:
        _file = open( "Attendance\\Attendance.csv" , "a" , newline='\n' )
        writer = csv.writer(_file , delimiter=',')
        # writer.writerow([ ID , NAME , DATE , TIME ])
        _info_write = [ str(_mark_my_attendance_profile[0]) , str(_mark_my_attendance_profile[1]) , str(day)+"-"+str(mont[month])+"-"+str(year) , str( time.strftime('%H:%M:%S') ) ]
        writer.writerow(_info_write)
        _file.close()

        cap.release()
        new_window.destroy()
        _mark_my_attendance_profile = None
        update_treeview()
        tk.messagebox.showinfo( "Status" , "Attendance Marked Successfully")
    else:
        tk.messagebox.showinfo( "Error" , "No Face Detected" , parent=new_window)

def get_profile_data():
    out = {}
    _file = open( "EmployeeDetails\\EmployeeDetails.csv" , newline='' )
    reader = csv.reader(_file , delimiter=',')

    for row in reader:
        out[row[0]] = row
    return out

def new_window_for_attendance(window):
    new_window = tk.Toplevel(window)
    new_window.geometry("700x550")
    new_window.resizable(False,False)
    new_window.title("Attendance")
    new_window.configure(background="white")

    label = tk.Label(new_window)
    label.grid(row=0, column=0)

    recognizer = cv2.face_LBPHFaceRecognizer.create()
    # recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("EmployeeDetails\\TrainedData\\trainedData.yml")

    cap = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier("files\haarcascade_frontalface_default.xml")

    all_profile_data = get_profile_data()
    # print(all_profile_data)

    takeImg = tk.Button(new_window, text="Mark My Attendance" , command= lambda : mark_my_attendance( cap , new_window ) ,fg="black"  ,bg="red"  ,width=34  ,height=1, activebackground = "white" ,font=('times', 15, ' bold '))
    takeImg.place(x=155, y=485)

    show_frames_for_attendance(cap, detector , recognizer, all_profile_data , label )

def take_attendance(window):
    new_window_for_attendance(window)

def get_employee_details():
    result = 0
    if os.path.isfile("EmployeeDetails\EmployeeDetails.csv"):
        with open("EmployeeDetails\EmployeeDetails.csv", 'r') as csvFile:
            reader = csv.reader(csvFile)
            for l in reader:
                result = result + 1
        result = result
        csvFile.close()
    return result

def view_all_window(window):
    new_window = tk.Toplevel(window)
    new_window.geometry("720x450")
    new_window.title("All Profiles")
    new_window.configure(background="white")

    # frame
    frame = tk.Frame(new_window)
    frame.place(relx=0.05, rely=0.05, relwidth=0.9, relheight=0.9)

    ################## TREEVIEW ATTENDANCE TABLE ####################
    tv= ttk.Treeview(frame,height =20,columns = ('id','name','date','time'))
    tv.column('#0',width=82)
    tv.column('id',width=100)
    tv.column('name',width=160)
    tv.column('date',width=143)
    tv.column('time',width=133)
    tv.grid(row=0,column=0,padx=(0,0),pady=(0,0),columnspan=4)
    tv.heading('#0',text ='SR. No.')
    tv.heading('id',text ='ID')
    tv.heading('name',text ='NAME')
    tv.heading('date',text ='DATE')
    tv.heading('time',text ='TIME')

    ###################### SCROLLBAR ################################
    scroll=ttk.Scrollbar(frame,orient='vertical',command=tv.yview)
    scroll.grid(row=0,column=4,padx=(0,0),pady=(0,0),sticky='ns')
    tv.configure(yscrollcommand=scroll.set)

    ## fill all the details
    if os.path.isfile("EmployeeDetails\EmployeeDetails.csv"):
        _file = open( "EmployeeDetails\EmployeeDetails.csv" , newline='' )
        reader = csv.reader(_file , delimiter=',')
        _idx = 1
        for row in reader:
            tv.insert('', 0, text=str(_idx), values=( row[0] , row[1] , row[2] , row[3] ))
            _idx += 1
        _file.close()



def main():
    window = tk.Tk()
    window.geometry("1280x720")
    window.resizable(True,False)
    window.title("Employee Attendance Management System")

    # preload colors
    _bg = "white"
    _text_color = "black"

    window.configure(background=_bg)

    # Header
    tk.Label(window, text="Employee Attendance Management System" ,fg=_text_color,bg=_bg ,width=55 ,height=1,font=('times', 29, ' bold ')).place(x=10, y=10)

    # Date and time Setup
    frame3 = tk.Frame(window, bg="#c4c6ce")
    frame3.place(relx=0.36, rely=0.09, relwidth=0.16, relheight=0.07)
    tk.Label(frame3, text = day+"-"+mont[month]+"-"+year+"  |  ", fg="orange",bg=_bg ,width=55 ,height=1,font=('times', 22, ' bold ')).pack(fill='both',expand=1)

    frame4 = tk.Frame(window, bg="#c4c6ce")
    frame4.place(relx=0.52, rely=0.09, relwidth=0.09, relheight=0.07)
    clock = tk.Label(frame4,fg="orange",bg=_bg ,width=55 ,height=1,font=('times', 22, ' bold '))
    clock.pack(fill='both',expand=1)
    tick(clock)  # This Updates the Clock every 200 ms.

    # frame 1 -> i.e - left part
    frame1 = tk.Frame(window, bg="#00aeff")
    frame1.place(relx=0.11, rely=0.17, relwidth=0.39, relheight=0.80)

    head1 = tk.Label(frame1, text="                       For Already Registered                       ", fg="black",bg="#03F2FD" ,font=('times', 17, ' bold ') )
    head1.place(x=0,y=0)

    lbl3 = tk.Label(frame1, text="Attendance",width=20  ,fg="black"  ,bg="#00aeff"  ,height=1 ,font=('times', 17, ' bold '))
    lbl3.place(x=100, y=115)

    takeImg = tk.Button(frame1, text="Take Attendance", command=lambda : take_attendance(window)  ,fg="black"  ,bg="#1AFD03"  ,width=35  ,height=1, activebackground = "white" ,font=('times', 15, ' bold '))
    takeImg.place(x=30,y=50)
    quitWindow = tk.Button(frame1, text="Quit", command=window.destroy  ,fg="black"  ,bg="red"  ,width=35 ,height=1, activebackground = "white" ,font=('times', 15, ' bold '))
    quitWindow.place(x=30, y=450)

    ################## TREEVIEW ATTENDANCE TABLE ####################
    tv= ttk.Treeview(frame1,height =13,columns = ('id','name','date','time'))
    tv.column('#0',width=40)
    tv.column('id',width=90)
    tv.column('name',width=100)
    tv.column('date',width=123)
    tv.column('time',width=123)
    tv.grid(row=2,column=0,padx=(0,0),pady=(150,0),columnspan=4)
    tv.heading('#0',text ='Sr. No.')
    tv.heading('id',text ='ID')
    tv.heading('name',text ='NAME')
    tv.heading('date',text ='DATE')
    tv.heading('time',text ='TIME')
    global _treeview
    _treeview = tv # Assigning it to global variable


    ###################### SCROLLBAR ################################
    scroll=ttk.Scrollbar(frame1,orient='vertical',command=tv.yview)
    scroll.grid(row=2,column=4,padx=(0,100),pady=(150,0),sticky='ns')
    tv.configure(yscrollcommand=scroll.set)


    # frame 2
    frame2 = tk.Frame(window, bg="#00aeff")
    frame2.place(relx=0.51, rely=0.17, relwidth=0.38, relheight=0.80)

    head2 = tk.Label(frame2, text="                       For New Registrations                        ", fg="black",bg="#03F2FD" ,font=('times', 17, ' bold ') )
    head2.grid(row=0,column=0)

    lbl = tk.Label(frame2, text="Enter Employee ID",width=20  ,height=1  ,fg="black"  ,bg="#00aeff" ,font=('times', 17, ' bold ') )
    lbl.place(x=80, y=55)

    txt = tk.Entry(frame2,width=32 ,fg="black",font=('times', 15, ' bold '))
    txt.place(x=30, y=88)

    lbl2 = tk.Label(frame2, text="Enter Employee Name",width=20  ,fg="black"  ,bg="#00aeff" ,font=('times', 17, ' bold '))
    lbl2.place(x=80, y=140)

    txt2 = tk.Entry(frame2,width=32 ,fg="black",font=('times', 15, ' bold ')  )
    txt2.place(x=30, y=173)

    message1 = tk.Label(frame2, text="1)Take Images  >>>  2)Save Profile" ,bg="#00aeff" ,fg="black"  ,width=39 ,height=1, activebackground = "yellow" ,font=('times', 15, ' bold '))
    message1.place(x=7, y=230)

    message = tk.Label(frame2, text="" ,bg="#00aeff" ,fg="black"  ,width=39,height=1, activebackground = "yellow" ,font=('times', 16, ' bold '))
    message.place(x=7, y=450)
    message.configure(text='Total Registrations till now  : '+str(get_employee_details()))

    clearButton = tk.Button(frame2, text="Clear", command=lambda : clear(txt)  ,fg="black"  ,bg="#ea2a2a"  ,width=11 ,activebackground = "white" ,font=('times', 11, ' bold '))
    clearButton.place(x=335, y=86)
    clearButton2 = tk.Button(frame2, text="Clear", command=lambda : clear(txt2)  ,fg="black"  ,bg="#ea2a2a"  ,width=11 , activebackground = "white" ,font=('times', 11, ' bold '))
    clearButton2.place(x=335, y=172)

    takeImg = tk.Button(frame2, text="Take Images", command=lambda : take_image(window, [str(txt.get()) , str(txt2.get())] )  ,fg="black"  ,bg="#F5FD03"  ,width=34  ,height=1, activebackground = "white" ,font=('times', 15, ' bold '))
    takeImg.place(x=30, y=300)
    trainImg = tk.Button(frame2, text="Save Profile", command=lambda : ( save_profile( [ txt , txt2 ]) , update(message) ) ,fg="black"  ,bg="#F5FD03"  ,width=34  ,height=1, activebackground = "white" ,font=('times', 15, ' bold '))
    trainImg.place(x=30, y=380)

    trainImg = tk.Button(frame2, text="View All Profiles", command=lambda : ( view_all_window(window) ) ,fg="black"  ,bg="#F5FD03"  ,width=34  ,height=1, activebackground = "white" ,font=('times', 15, ' bold '))
    trainImg.place(x=30, y=480)

    update_treeview()
    window.mainloop()

# train_images()
main()
