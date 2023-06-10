from flask import (
    Flask,
    render_template,
    request,
    session,
    redirect,
    url_for,
    Response,
    jsonify,
)
import mysql.connector
import cv2
from PIL import Image
import numpy as np
import os
import time
from datetime import date

app = Flask(__name__)
cnt = 0
pause_cnt = 0
justscanned = False
mydb = mysql.connector.connect(
    host="localhost", user="root", passwd="", database="nhandienkhuonmat"
)
mycursor = mydb.cursor()


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Generate dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def generate_dataset(nbr):
    face_classifier = cv2.CascadeClassifier(
        "resources/haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)
    mycursor.execute("select ifnull(max(img_id), 0) from img_dataset")
    row = mycursor.fetchone()
    lastid = row[0] + 1
    img_id = lastid
    max_imgid = img_id + 50
    count_img = 0

    while True:
        ret, img = cap.read()
        imgDetection = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2 = face_classifier.detectMultiScale(
            img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        for fX, fY, fW, fH in img2:
            if count_img < 50:
                cv2.rectangle(img, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
                file_name_path = "dataset/" + nbr + "." + str(img_id) + ".jpg"
                cv2.imwrite(file_name_path, imgDetection)
                cv2.putText(
                    img,
                    str(count_img),
                    (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                mycursor.execute(
                    """INSERT INTO `img_dataset` (`img_id`, `img_sv`) VALUES
                                    ('{}', '{}')""".format(
                        img_id, nbr
                    )
                )
                mydb.commit()
                count_img += 1
                img_id += 1
            else:
                return

        frame = cv2.imencode(".jpg", img)[1].tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        if cv2.waitKey(1) == 13 or int(img_id) == int(max_imgid):
            cap.release()
            cv2.destroyAllWindows()
            break


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Train Classifier >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route("/train_classifier/<nbr>")
def train_classifier(nbr):
    dataset_dir = "dataset"
    path = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    faces = []
    ids = []
    for image in path:
        img = Image.open(image).convert("L")
        imageNp = np.array(img, "uint8")
        id = int(os.path.split(image)[1].split(".")[1])
        faces.append(imageNp)
        ids.append(id)
    ids = np.array(ids)
    # Train the classifier and save
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")

    return redirect("/student_page")


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Face Recognition >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def face_recognition(idsubject, classid):  # generate frame by frame from camera
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)
        global justscanned
        global pause_cnt
        pause_cnt += 1
        coords = []
        for x, y, w, h in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            id, pred = clf.predict(gray_image[y : y + h, x : x + w])
            confidence = int(100 * (1 - pred / 300))
            if confidence > 70 and not justscanned:
                global cnt
                cnt += 1

                n = (100 / 30) * cnt
                # w_filled = (n / 100) * w
                w_filled = (cnt / 30) * w

                cv2.putText(
                    img,
                    str(int(n)) + " %",
                    (x + 20, y + h + 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (153, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.rectangle(img, (x, y + h + 40), (x + w, y + h + 50), color, 2)
                cv2.rectangle(
                    img,
                    (x, y + h + 40),
                    (x + int(w_filled), y + h + 50),
                    (153, 255, 255),
                    cv2.FILLED,
                )

                mycursor.execute(
                    "select a.img_sv, b.sv_name, b.sv_class "
                    "  from img_dataset a "
                    "  left join student b on a.img_sv = b.mssv "
                    " where img_id = " + str(id)
                )
                row = mycursor.fetchone()
                pnbr = row[0]
                pname = row[1]
                pskill = row[2]

                if int(cnt) == 30:
                    cnt = 0

                    mycursor.execute(
                        # "insert into accs_hist (accs_date, accs_prsn,dateID) values('"
                        # + str(date.today())
                        # + "', '"
                        # + pnbr
                        # + "')"
                        "SELECT * FROM `accs_hist` WHERE  student_id='{}' and class_module_id='{}'".format(
                            pnbr, classid
                        )
                    )
                    data = mycursor.fetchone()
                    if len(data) == 0:
                        mycursor.execute(
                            # "insert into accs_hist (accs_date, accs_prsn,dateID) values('"
                            # + str(date.today())
                            # + "', '"
                            # + pnbr
                            # + "')"
                            "insert into accs_hist (accs_date, student_id,dateID,class_module_id) values('{}','{}','{}','{}')".format(
                                str(date.today()), pnbr, idsubject, classid
                            )
                        )
                        mydb.commit()

                    cv2.putText(
                        img,
                        pname + " | " + pskill,
                        (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (153, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    time.sleep(1)

                    justscanned = True
                    pause_cnt = 0

            else:
                if not justscanned:
                    cv2.putText(
                        img,
                        "UNKNOWN",
                        (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.putText(
                        img,
                        " ",
                        (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                if pause_cnt > 80:
                    justscanned = False

            coords = [x, y, w, h]
        return coords

    def recognize(img, clf, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 0), "Face", clf)
        return img

    faceCascade = cv2.CascadeClassifier("resources/haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")
    wCam, hCam = 400, 400
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    while True:
        ret, img = cap.read()

        img = recognize(img, clf, faceCascade)
        frame = cv2.imencode(".jpg", img)[1].tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")
        key = cv2.waitKey(1)
        if key == 27:
            break


@app.route("/student_page")
def home():
    mycursor.execute("select mssv, sv_name, sv_class from student")
    data = mycursor.fetchall()
    return render_template("student_page.html", data=data)


@app.route("/sv", methods=["GET"])
def sv():
    mycursor.execute("select mssv, sv_name, sv_class from student")
    data = mycursor.fetchall()
    return jsonify(response=data)


@app.route("/addprsn")
def addprsn():
    mycursor.execute("select ifnull(max(mssv) + 0, 0)  from student")
    row = mycursor.fetchone()
    nbr = row[0]
    # print(int(nbr))

    return render_template("add_student_page.html", newnbr=int(nbr))


@app.route("/addprsn_submit", methods=["POST"])
def addprsn_submit():
    mssv = request.form.get("mssv")
    namesv = request.form.get("txtname")
    classsv = request.form.get("class")
    sql = "SELECT * FROM `student` WHERE mssv ='{}'".format(mssv)
    mycursor.execute(sql)
    data = mycursor.fetchall()
    if len(data) > 0:
        return redirect(url_for("addprsn"))
    else:
        mycursor.execute(
            """INSERT INTO `student` (`mssv`, `sv_name`, `sv_class`) VALUES
                        ('{}', '{}', '{}')""".format(
                mssv, namesv, classsv
            )
        )
        mydb.commit()
        return redirect(url_for("gendataset_page", prs=mssv))


@app.route("/vidfeed_dataset/<nbr>")
def vidfeed_dataset(nbr):
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(
        generate_dataset(nbr), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/video_feed/<id>/<classid>")
def video_feed(id, classid):
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(
        face_recognition(id, classid),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/fr_page/<id>/<classid>")
def fr_page(id, classid):
    """Video streaming home page."""
    mycursor.execute(
        "select a.accs_id, a.student_id, b.sv_name, b.sv_class, a.accs_added "
        "  from accs_hist a "
        "  left join student b on a.student_id = b.mssv "
        " where a.dateID='{}' and a.class_module_id='{}'".format(id, classid)
    )
    data = mycursor.fetchall()

    return render_template("fr_page.html", data=data, id=id, classid=classid)


@app.route("/countTodayScan/<id>/<classid>")
def countTodayScan(id, classid):
    mydb = mysql.connector.connect(
        host="localhost", user="root", passwd="", database="nhandienkhuonmat"
    )
    mycursor = mydb.cursor()

    mycursor.execute(
        "select count(*)  from accs_hist where dateID ='{}' and class_module_id='{}' ".format(
            id, classid
        )
    )
    row = mycursor.fetchone()
    rowcount = row[0]

    return jsonify({"rowcount": rowcount})


@app.route("/loadData/<id>/<classid>", methods=["GET", "POST"])
def loadData(id, classid):
    mydb = mysql.connector.connect(
        host="localhost", user="root", passwd="", database="nhandienkhuonmat"
    )
    mycursor = mydb.cursor()

    mycursor.execute(
        "select a.accs_id, a.student_id, b.sv_name, b.sv_class, date_format(a.accs_added, '%H:%i:%s') "
        "  from accs_hist a "
        " left join student b on a.student_id = b.mssv "
        " where a.dateID ='{}' and a.class_module_id='{}' "
        " order by 1 desc".format(id, classid)
    )
    data = mycursor.fetchall()
    print(data, "2")
    return jsonify(response=data)


@app.route("/gendataset_page/<prs>")
def gendataset_page(prs):
    return render_template("gendataset_page.html", prs=prs)


@app.route("/add_class_module")
def addclassmodule_page():
    mycursor.execute("select * from class_module")
    data = mycursor.fetchall()

    return render_template("classmodule_page.html", classmodules=data)


@app.route("/add_class_page")
def add_class_page():
    return render_template("add_class_page.html")


@app.route("/class_module")
def classmodule_page():
    mycursor.execute("select * from class_module")
    data = mycursor.fetchall()

    return render_template("classmodule_page.html", classmodules=data)


@app.route("/class_module_page/add", methods=["POST"])
def addclassmodule():
    try:
        classmodule_name = request.form.get("classmodule_name")
        lecture_name = request.form.get("lecture_name")

        mycursor.execute(
            """INSERT INTO `class_module` (`name`, `lecturer_name`) VALUES
                        ('{}', '{}')""".format(
                classmodule_name, lecture_name
            )
        )
        mydb.commit()
        return redirect(url_for("classmodule_page", res="add_success"))
    except NameError:
        return redirect(url_for("classmodule_page", res="add_fail"))
    # return redirect(url_for('home'))


@app.route("/class_module_page/addstudent/<id>", methods=["POST"])
def addstudent(id):
    mssv = request.form.get("mssv")
    print("masv", mssv, id)
    mycursor.execute(
        """INSERT INTO `class_module_registration` (`class_module_id`, `student_id`) VALUES
                    ('{}', '{}')""".format(
            id, mssv
        )
    )
    mydb.commit()
    return redirect(url_for("student_register", id=id, success=True))


@app.route("/student_register/<id>/<success>")
def student_register(id, success):
    mycursor.execute(
        "SELECT * FROM class_module_registration  LEFT JOIN student on class_module_registration.student_id=student.mssv where class_module_registration.class_module_id='{}'".format(
            id
        )
    )
    data = mycursor.fetchall()

    mycursor.execute(
        "SELECT * FROM `student` WHERE mssv not in('{}')".format(
            "','".join(x[2] for x in data)
        )
    )
    print(",".join(x[2] for x in data))
    dataall = mycursor.fetchall()
    return render_template(
        "student_register.html", students=data, id=id, success=success, dataall=dataall
    )


@app.route("/subject/<id>")
def subject(id):
    mycursor.execute("SELECT * FROM `date_module` WHERE classID='{}'".format(id))
    data = mycursor.fetchall()
    mycursor.execute("SELECT * FROM `class_module` ")
    dataall = mycursor.fetchall()
    return render_template("subject.html", subjects=data, id=id, dataall=dataall)


@app.route("/detail/<dateid>/<id>")
def detail(dateid, id):
    mycursor.execute(
        "SELECT *,student.sv_name,student.sv_class FROM accs_hist left join student on student.mssv=accs_hist.student_id WHERE dateId='{}' and class_module_id='{}' ".format(
            dateid, id
        )
    )
    data = mycursor.fetchall()
    print("detail", data)
    return render_template("detail.html", students=data)


@app.route("/remove/<idstudent>/<idclass>")
def remove(idstudent, idclass):
    mycursor.execute(
        "DELETE FROM `class_module_registration` WHERE class_module_id='{}' and student_id='{}' ".format(
            idclass, idstudent
        )
    )
    data = mycursor.fetchall()

    return redirect(url_for("student_register", id=idclass, success=True))


@app.route("/addsubject", methods=["POST"])
def addsubject():
    subject = request.form.get("subject")
    classid = request.form.get("classid")
    mycursor.execute(
        "INSERT INTO `date_module`(  `classID`, `title`) VALUES ('{}','{}') ".format(
            classid, subject
        )
    )
    mycursor.fetchall()
    return redirect(url_for("subject", id=classid))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
