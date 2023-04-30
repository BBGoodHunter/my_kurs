import re
import shutil
import sys
import cv2 as cv
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.Qt import *
from pygrabber.dshow_graph import FilterGraph
import os
import MainWindow

block_vid = False
cut_cam = False
check_save_img = False
check_save_vid = False
screenshot_on = False
rec_status = False
search_word = 'all'
text_color = [0, 255, 0]
border_color = [0, 255, 0]
class_names1 = []
is_learn_on = False
block_cam = 1
get_camera = 0
file_name, file_type = '', ''
change_cam = 0

modelConfiguration = "yolov3-320.cfg"
modelWeights = "yolov3-320.weights"


class ThreadOpenCV(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()

    def run(self):
        global class_names1, block_cam, get_camera, file_name, file_type
        classes_file = "coco.names"
        class_names = []
        with open(classes_file, 'rt') as f:
            class_names = [line.rstrip() for line in f]
        class_names1 = sorted(class_names)

        net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        while True:
            if block_cam == 0:
                cap = cv.VideoCapture(get_camera)

            elif file_type == ".mp4":
                cap = cv.VideoCapture(file_name + file_type)

            elif file_type == ".jpg":
                cap = cv.VideoCapture(file_name + file_type)

            else:
                continue

            wh_t = 320
            conf_threshold = 0.5
            nms_threshold = 0.2

            def find_objects(outputs, img):
                global border_color, text_color, search_word
                h_t, w_t, c_t = img.shape
                bbox = []
                class_ids = []
                confs = []
                for output in outputs:
                    for det in output:
                        scores = det[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > conf_threshold:
                            w, h = int(det[2] * w_t), int(det[3] * h_t)
                            x, y = int((det[0] * w_t) - w / 2), int((det[1] * h_t) - h / 2)
                            bbox.append([x, y, w, h])
                            class_ids.append(class_id)
                            confs.append(float(confidence))

                indices = cv.dnn.NMSBoxes(bbox, confs, conf_threshold, nms_threshold)

                for i in indices:
                    box = bbox[i]
                    x, y, w, h = box[0], box[1], box[2], box[3]

                    if search_word == class_names[class_ids[i]] or search_word == 'all':
                        cv.rectangle(img, (x, y), (x + w, y + h), (border_color[0], border_color[1], border_color[2]), 2)
                        cv.putText(img, f'{class_names[class_ids[i]].upper()} {int(confs[i] * 100)}%',
                                   (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                                   (text_color[0], text_color[1], text_color[2]), 2)

            while True:
                success, img = cap.read()
                global check_save_vid, rec_status, cut_cam
                if not success or cut_cam:
                    if check_save_vid or cut_cam:
                        path = 'Temp/'
                        i = 1
                        while True:

                            image_name = "video"
                            path1 = 'Videos/' + image_name + str(i) + ".mp4"
                            if os.path.isfile(path1):
                                i += 1
                            else:
                                image_name = 'Videos/' + image_name + str(i) + ".mp4"
                                break
                        out_video_full_path = image_name

                        temp = os.listdir(path)
                        numre = re.compile('[0-9]+')

                        def extract_num(s):
                            return int(numre.search(s).group())

                        temp.sort(key=extract_num)
                        img = []

                        for i in temp:
                            i = path + i

                            img.append(i)

                        cv2_fourcc = cv.VideoWriter_fourcc(*'mp4v')

                        frame = cv.imread(img[0])
                        size = list(frame.shape)
                        del size[2]
                        size.reverse()

                        if file_type == ".mp4" or modelConfiguration == "yolov3-tiny.cfg":
                            video = cv.VideoWriter(out_video_full_path, cv2_fourcc, 24,
                                                   size)  # output video name, fourcc, fps, size
                        else:
                            video = cv.VideoWriter(out_video_full_path, cv2_fourcc, 2,
                                                   size)  # output video name, fourcc, fps, size
                        for i in range(len(img)):
                            video.write(cv.imread(img[i]))

                        video.release()
                        shutil.rmtree('Temp/')
                        cut_cam = False
                    if file_type == ".mp4":
                        global block_vid
                        block_vid = False
                    file_type = ''
                    break
                blob = cv.dnn.blobFromImage(img, 1 / 255, (wh_t, wh_t), [0, 0, 0], 1, crop=False)
                net.setInput(blob)
                layers_names = net.getLayerNames()
                output_names = [(layers_names[i - 1]) for i in net.getUnconnectedOutLayers()]
                outputs = net.forward(output_names)
                find_objects(outputs, img)
                if success:
                    rgb_image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    convert_to_qt_format = QImage(
                        rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    p = convert_to_qt_format.scaled(550, 550, Qt.KeepAspectRatio)
                    self.changePixmap.emit(p)
                    if file_type == '.jpg' and block_cam == 0:
                        block_cam = 1
                        break
                    if file_type == '.jpg' and block_cam == 1:
                        global check_save_img
                        if check_save_img:
                            if not (os.path.exists("Images")):
                                os.mkdir("Images")
                            i = 1
                            while True:

                                image_name = "image"
                                path = 'Images/' + image_name + str(i) + ".jpg"
                                if os.path.isfile(path):
                                    i += 1
                                else:
                                    image_name = 'Images/' + image_name + str(i) + ".jpg"
                                    convert_to_qt_format.save(image_name)
                                    break
                        file_type = ''
                        break
                    if file_type == '.mp4' and block_cam == 0:
                        block_cam = 1
                        break
                    global rec_status
                    if (file_type == '.mp4' and block_cam == 1) or rec_status:
                        if not check_save_vid and not rec_status:
                            continue
                        if not (os.path.exists("Temp")):
                            os.mkdir("Temp")
                        i = 1
                        while True:

                            path = 'Temp/' + str(i) + ".jpg"
                            if os.path.isfile(path):
                                i += 1
                            else:
                                image_name = 'Temp/' + str(i) + ".jpg"
                                convert_to_qt_format.save(image_name)
                                break

                    global screenshot_on
                    if screenshot_on:
                        if not (os.path.exists("Screenshots")):
                            os.mkdir("Screenshots")
                        i = 1
                        while True:

                            image_name = "screenshot"
                            path = 'Screenshots/' + image_name + str(i) + ".jpg"
                            if os.path.isfile(path):
                                i += 1
                            else:
                                image_name = 'Screenshots/' + image_name + str(i) + ".jpg"
                                convert_to_qt_format.save(image_name)
                                screenshot_on = False
                                break
                    global change_cam
                    if change_cam:
                        cap = cv.VideoCapture(get_camera)
                        change_cam = 0
                self.msleep(20)


class MainWindow(QtWidgets.QMainWindow, MainWindow.Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent=parent)
        self.setupUi(self)
        devices = FilterGraph().get_input_devices()
        self.comboBox_choosecam.activated.connect(self.get_cam)
        for device_index, device_name in enumerate(devices):
            self.comboBox_choosecam.addItem(device_name)
        self.comboBox_search.activated.connect(self.search_info)
        self.pushButton_beginlearn.clicked.connect(self.can)
        self.comboBox_chstextclr.activated.connect(lambda: self.change_color(1))
        self.comboBox_chsborderclr.activated.connect(lambda: self.change_color(2))
        # +++
        self.pushButton_savecam.clicked.connect(self.begin_rec)
        self.thread = ThreadOpenCV()  # +++
        self.pushButton_openimg.clicked.connect(self.upload_img)
        self.pushButton_openvid.clicked.connect(self.upload_vid)
        self.checkBox1.stateChanged.connect(self.check_vid)
        self.checkBox2.stateChanged.connect(self.check_img)
        self.pushButton_fast.clicked.connect(self.fast_learn)
        self.pushButton_accurate.clicked.connect(self.accurate_learn)
        self.pushButton_screenshot.clicked.connect(self.take_screen)
        self.thread.changePixmap.connect(self.set_image)  # +++
        if os.path.exists("Temp/"):
            shutil.rmtree('Temp/')

    def begin_rec(self):
        if not is_learn_on:
            self.err_msg_learn()
            return
        global block_vid
        if block_vid:
            self.err_msg_video()
            return
        global block_cam
        if block_cam == 1:
            msg = QMessageBox()
            msg.setWindowTitle("Ошибка")
            msg.setText("Выберите камеру для записи.")
            msg.setIcon(QMessageBox.Warning)
            msg.exec_()
            return
        global rec_status, cut_cam
        if not rec_status:
            self.pushButton_savecam.setText("Остановить запись")
            self.label_current_status.setText(
                f"Статус: Идет запись анализа с веб-камеры {self.comboBox_choosecam.currentText()}")
            rec_status = True
        else:
            cut_cam = True
            self.pushButton_savecam.setText("Начать запись")
            self.label_current_status.setText(
                f"Статус: Анализ веб-камеры {self.comboBox_choosecam.currentText()}")
            rec_status = False

    def check_img(self):
        if self.checkBox2.isChecked():
            global check_save_img
            check_save_img = True
        else:
            check_save_img = False

    def check_vid(self):
        if self.checkBox1.isChecked():
            global check_save_vid
            check_save_vid = True
        else:
            check_save_vid = False

    def take_screen(self):
        if not is_learn_on:
            self.err_msg_learn()
            return
        global screenshot_on
        screenshot_on = True

    def search_info(self):
        global class_names1, search_word
        if self.comboBox_search.count() > 2:
            temp = self.comboBox_search.currentIndex()
            if temp == 0:
                search_word = 'all'
                return
            search_word = class_names1[temp - 1]
        else:
            for (index, elem) in enumerate(class_names1):
                self.comboBox_search.addItem(elem)

    def change_color(self, mode):

        if mode == 1:
            index = self.comboBox_chstextclr.currentIndex()
            global text_color
            if index == 0:
                text_color = [0, 255, 0]
            if index == 1:
                text_color = [0, 0, 255]
            if index == 2:
                text_color = [255, 0, 0]
        if mode == 2:
            index = self.comboBox_chsborderclr.currentIndex()
            global border_color
            if index == 0:
                border_color = [0, 255, 0]
            if index == 1:
                border_color = [0, 0, 255]
            if index == 2:
                border_color = [255, 0, 0]

    def get_cam(self):
        global block_vid, block_cam
        if block_vid:
            self.err_msg_video()
            return
        global is_learn_on
        if not is_learn_on:
            self.err_msg_learn()
            return
        if block_cam == 0:
            global change_cam
            change_cam = 1
        get_index_cam = self.comboBox_choosecam.currentIndex()
        if get_index_cam != 0:
            global get_camera
            get_camera = get_index_cam - 1
            self.label_current_status.setText(f"Статус: Анализ веб-камеры {self.comboBox_choosecam.currentText()}")
            block_cam = 0

    @staticmethod
    def fast_learn():
        global is_learn_on
        if is_learn_on:
            msg = QMessageBox()
            msg.setWindowTitle("Ошибка")
            msg.setText("Обучение уже начато.")
            msg.setIcon(QMessageBox.Warning)
            msg.exec_()
        else:
            global modelConfiguration
            global modelWeights
            modelConfiguration = "yolov3-tiny.cfg"
            modelWeights = "yolov3-tiny.weights"

    @staticmethod
    def accurate_learn():
        global is_learn_on
        if is_learn_on:
            msg = QMessageBox()
            msg.setWindowTitle("Ошибка")
            msg.setText("Обучение уже начато.")
            msg.setIcon(QMessageBox.Warning)
            msg.exec_()
        else:
            global modelConfiguration
            global modelWeights
            modelConfiguration = "yolov3-320.cfg"
            modelWeights = "yolov3-320.weights"

    def upload_img(self):
        global block_vid
        if block_vid:
            self.err_msg_video()
            return
        global is_learn_on
        if not is_learn_on:
            self.err_msg_learn()
            return
        global file_name, file_type, block_cam
        temp1, temp = QFileDialog.getOpenFileName(
            self,
            "Open file",
            ".",
            "Image Files (*.jpg)"
        )
        file_name, file_type = os.path.splitext(temp1)
        self.label_current_status.setText(f"Статус: Анализ изображения {file_name + file_type}")

    def upload_vid(self):
        global block_vid
        if block_vid:
            self.err_msg_video()
            return
        global is_learn_on
        if not is_learn_on:
            self.err_msg_learn()
            return
        global file_name, file_type
        temp1, temp = QFileDialog.getOpenFileName(
            self,
            "Open file",
            ".",
            "Video Files (*.mp4)"
        )
        file_name, file_type = os.path.splitext(temp1)
        block_vid = True
        self.label_current_status.setText(f"Статус: Анализ видео {file_name + file_type}")

    @staticmethod
    def err_msg_learn():
        msg = QMessageBox()
        msg.setWindowTitle("Ошибка")
        msg.setText("Обучение уже начато.")
        msg.setIcon(QMessageBox.Warning)
        msg.exec_()

    @staticmethod
    def err_msg_video():
        msg = QMessageBox()
        msg.setWindowTitle("Ошибка")
        msg.setText("Дождитесь окончания обработки видео.")
        msg.setIcon(QMessageBox.Warning)
        msg.exec_()

    def can(self):

        global is_learn_on
        if is_learn_on:
            msg = QMessageBox()
            msg.setWindowTitle("Ошибка")
            msg.setText("Обучение уже начато.")
            msg.setIcon(QMessageBox.Warning)
            msg.exec_()
        else:
            is_learn_on = True
            model_type = ''
            if modelConfiguration == "yolov3-320.cfg":
                model_type = 'YOLOv3-320'
            if modelConfiguration == "yolov3-tiny.cfg":
                model_type = 'YOLOv3-tiny'
            self.label_current_status.setText(f"Статус: Обучение прошло успешно, модель обучения {model_type}")
            self.thread.start()  # +++

    def set_image(self, image):  # +++
        self.label_video.setPixmap(QPixmap.fromImage(image))  # +++


if __name__ == "__main__":
    while True:
        app = QtWidgets.QApplication(sys.argv)
        w = MainWindow()
        w.show()
        sys.exit(app.exec_())
