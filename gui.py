import sys
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import image_cal
import graph_cal
import text_cal


def cal_percent(rank):
    num_list = list(map(int,rank.split('/')))
    return num_list[0]/num_list[1]*100
class MyApp(QMainWindow):

    def __init__(self, namelist, f_ext):
        super().__init__()
        self.nl = namelist
        self.initUI()
        self.f_ext = f_ext
        self.data_base = []
        self.data_base2 = {}


    def initUI(self):
        self.setWindowTitle('MUDASA(Multi-modal Unstructured Data based Architecture Similarity Assessment)')
        self.setWindowIcon(QIcon('./background/icon.png'))
        self.resize(1800, 1600)
        self.center()

        menubar = self.menuBar()
        actionFile = menubar.addMenu("File")
        actionFile.addAction("New")
        actionFile.addAction("Open")
        actionFile.addAction("Save")
        actionFile.addSeparator()
        actionFile.addAction("Quit")
        menubar.addMenu("Edit")
        menubar.addMenu("View")
        menubar.addMenu("Help")

        #### ProgressBar 추가하기
        self.pbar = QProgressBar(self)
        self.pbar.setMinimum(0)
        self.pbar.setMaximum(100)

        self.label = QLabel() ### 그래프 사진
        self.label2 = QLabel() ### 외관사진
        self.label3 = QLabel() ### 내관사진
        self.label4 = QLabel() ### 도면사진
        self.label5 = QLabel() ### 텍스트파일
        self.label6 = QLabel() ### 토폴로지파일
        self.label7 = QLabel() ### 보고서내용

        self.label7.setAlignment(Qt.AlignCenter)

        self.fl1 = QLabel()
        self.fl2 = QLabel()
        self.fl3 = QLabel()
        self.fl4 = QLabel()


        layout = QGridLayout()


        self.pushButton = QtWidgets.QPushButton("Upload Exterior Images")
        self.pushButton.clicked.connect(self.exterior_fileopen)
        self.pushButton.setCheckable(True)

        self.pushButton2 = QtWidgets.QPushButton("Upload Interior Images")
        self.pushButton2.clicked.connect(self.interior_fileopen)
        self.pushButton2.setCheckable(True)

        self.pushButton3 = QtWidgets.QPushButton("Upload Plan Images")
        self.pushButton3.clicked.connect(self.graph_fileopen)
        self.pushButton3.setCheckable(True)

        self.pushButton4 = QtWidgets.QPushButton("Upload Concept Texts")
        self.pushButton4.clicked.connect(self.text_fileopen)
        self.pushButton4.setCheckable(True)

        self.pushButton6 = QtWidgets.QPushButton("Upload Graph Files")
        self.pushButton6.clicked.connect(self.topo_fileopen)
        self.pushButton6.setCheckable(True)

        self.pushButton5 = QtWidgets.QPushButton("Calculation Similarity")
        self.pushButton5.clicked.connect(self.calculation)

        ### 내관 외관 플랜 버튼 배치
        layout.addWidget(self.pushButton,0,0,1,10)

        layout.addWidget(self.pushButton2,0,10,1,10)

        layout.addWidget(self.pushButton3,0,20,1,10)

        ### 컨셉 공간 버튼 배치
        layout.addWidget(self.pushButton4,1,0,1,15)
        layout.addWidget(self.pushButton6,1,15,1,15)

        ##### 완료표시 넣어주기
        layout.addWidget(self.label2,2,0,1,6)
        layout.addWidget(self.label3,2,6,1,6)
        layout.addWidget(self.label4,2,12,1,6)
        layout.addWidget(self.label5,2,18,1,6)
        layout.addWidget(self.label6,2,24,1,6)

        #### 버튼엔 라벨 넣어주기
        layout.addWidget(self.pushButton5, 3, 0, 1, 30)

        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.pbar,4,0,1, 30)
        layout.addWidget(self.label,5,0,5,30)
        layout.addWidget(self.label7, 6, 0, 8, 30)

        layout.addWidget(self.fl1, 13,1, 3, 7)
        layout.addWidget(self.fl2, 13,8, 3, 7)
        layout.addWidget(self.fl3, 13,15, 3, 7)
        layout.addWidget(self.fl4, 13,22, 3, 7)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)


        self.show()


    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def exterior_fileopen(self):
        self.nl['Exterior'] = []
        self.nl['Exterior'].append(QtWidgets.QFileDialog.getOpenFileName(self)[0])
        self.nl['Exterior'].append(QtWidgets.QFileDialog.getOpenFileName(self)[0])
        if len(self.nl['Exterior']) <2:
            self.label2.setText('외관사진을 다시 업로드해주세요')
        else:
            self.label2.setText('Exterior Images Upload Complete!')
    def interior_fileopen(self):
        self.nl['Interior'] = []
        self.nl['Interior'].append(QtWidgets.QFileDialog.getOpenFileName(self)[0])
        self.nl['Interior'].append(QtWidgets.QFileDialog.getOpenFileName(self)[0])
        if len(self.nl['Interior']) < 2:
            self.label3.setText('내관사진을 다시 업로드해주세요')
        else:
            self.label3.setText('Interior Images Upload Complete!')

    def graph_fileopen(self):
        self.nl['Topology(Plan)'] = []
        self.nl['Topology(Plan)'].append(QtWidgets.QFileDialog.getOpenFileName(self)[0])
        self.nl['Topology(Plan)'].append(QtWidgets.QFileDialog.getOpenFileName(self)[0])
        if len(self.nl['Topology(Plan)']) < 2:
            self.label4.setText('도면사진을 다시 업로드해주세요')
        else:
            self.label4.setText('Plan Images Upload Complete!')

    def text_fileopen(self):
        self.nl['Concept'] = []
        self.nl['Concept'].append(QtWidgets.QFileDialog.getOpenFileName(self)[0])
        self.nl['Concept'].append(QtWidgets.QFileDialog.getOpenFileName(self)[0])
        if len(self.nl['Concept']) < 2:
            self.label5.setText('컨셉글을 다시 업로드해주세요')
        else:
            self.label5.setText('Concept Upload Complete!')

    def topo_fileopen(self):
        self.nl['Topology(Connection)'] = []
        self.nl['Topology(Connection)'].append(QtWidgets.QFileDialog.getOpenFileName(self)[0])
        self.nl['Topology(Connection)'].append(QtWidgets.QFileDialog.getOpenFileName(self)[0])
        if len(self.nl['Topology(Connection)']) < 2:
            self.label6.setText('위상정보를 다시 업로드해주세요')
        else:
            self.label6.setText('Graph Upload Complete!')

    def calculation(self):
        self.data_base2 = {}

        print(self.nl)
        self.pbar.setValue(10)
        self.label.setText('유사도 계산중')

        if len(self.nl['Exterior'][0]) + len(self.nl['Exterior'][1]) >5:
            a,b,c,d,e,f = image_cal.image_calculation(self.nl['Exterior'][0], self.nl['Exterior'][1], 'exterior', self.f_ext)
            print(a,b,c)
            self.data_base.append(c)
            self.data_base2['Exterior'] = f
        else:
            self.label.setText('올바른 유사도 비교쌍이 생성되지 않았습니다')
        self.pbar.setValue(30)
        if len(self.nl['Interior'][0]) + len(self.nl['Interior'][1]) >5:
            a,b,c,d,e,f = image_cal.image_calculation(self.nl['Interior'][0], self.nl['Interior'][1], 'interior', self.f_ext)
            print(a,b,c)
            self.data_base.append(c)
            self.data_base2['Interior'] = f
        else:
            self.label.setText('올바른 유사도 비교쌍이 생성되지 않았습니다')
        self.pbar.setValue(50)
        if len(self.nl['Topology(Plan)'][0]) + len(self.nl['Topology(Plan)'][1]) >5:
            a,b,c,d,e,f = image_cal.image_calculation(self.nl['Topology(Plan)'][0], self.nl['Topology(Plan)'][1], 'plan', self.f_ext)

            self.data_base.append(c)
            self.data_base2['Topology(Plan)'] = f
        else:
            self.label.setText('올바른 유사도 비교쌍이 생성되지 않았습니다')
        self.pbar.setValue(70)
        if len(self.nl['Concept'][0]) + len(self.nl['Concept'][1]) > 5:
            a, b,c,d = text_cal.text_calculation(self.nl['Concept'][0], self.nl['Concept'][1])

            self.data_base.append(a)
            self.data_base2['Concept'] = c
        else:
            self.label.setText('올바른 유사도 비교쌍이 생성되지 않았습니다')
        self.pbar.setValue(90)
        if len(self.nl['Topology(Connection)'][0]) + len(self.nl['Topology(Connection)'][1]) >5:
            a,b = graph_cal.graph_calculation(self.nl['Topology(Connection)'][0], self.nl['Topology(Connection)'][1])

            self.data_base.append(a)
            self.data_base2['Topology(Connection)'] = b
        else:
            self.label.setText('올바른 유사도 비교쌍이 생성되지 않았습니다')

        import draw_graph
        self.data_base.append(self.data_base[0])
        draw_graph.drawG(self.data_base, 'A','B')

        ### 사진넣어주기!!!
        pixmap = QPixmap('./temp_result/A&B.png')
        self.label.setPixmap(pixmap)
        temp1 = cal_percent(self.data_base2['Exterior'])
        temp2 = cal_percent(self.data_base2['Interior'])
        temp3 = cal_percent(self.data_base2['Topology(Plan)'])
        temp4 = cal_percent(self.data_base2['Concept'])
        temp5 = cal_percent(self.data_base2['Topology(Connection)'])
        tt_temp = {}
        tt_temp['Exterior'] = temp1
        tt_temp['Interior'] = temp2
        tt_temp['Topology(Plan)'] = temp3
        tt_temp['Concept'] = temp4
        tt_temp['Topology(Connection)'] = temp5
        tt_temp = sorted(tt_temp.items(), key = lambda x : x[1])
        self.label7.setText(f'The similarity analysis details are as follows.\nBuilding A and Building B\'s Exterior Similarity is in the top {temp1:.3f}% of 947 comparison pairs,\nInterior Similarity is in the top {temp2:.3f}% of 947 comparison pairs,\n'
                            f'Topology(Plan) Similarity is in the top {temp3:.3f}% of 947 comparison paris,\nTopology(Connection) Similairty is in the top {temp5:.3f}% of 947 comparisons,\nConcept Similarity is in the top {temp4:.3f}% of 947 comparison pairs.\n'
                            f'If plagiarism is suspected, It is need to review it mainly by {tt_temp[0][0]}, {tt_temp[1][0]}.')

        if tt_temp[0][0] == 'Exterior':
            pixmap = QPixmap(self.nl['Exterior'][0])
            self.fl1.setPixmap(pixmap.scaled(self.fl1.size(), Qt.IgnoreAspectRatio))
            pixmap = QPixmap(self.nl['Exterior'][1])
            self.fl2.setPixmap(pixmap.scaled(self.fl2.size(),Qt.IgnoreAspectRatio))

        elif tt_temp[0][0] == 'Interior':
            pixmap = QPixmap(self.nl['Interior'][0])
            self.fl1.setPixmap(pixmap.scaled(self.fl1.size(),Qt.IgnoreAspectRatio))
            pixmap = QPixmap(self.nl['Interior'][1])
            self.fl2.setPixmap(pixmap.scaled(self.fl2.size(),Qt.IgnoreAspectRatio))

        elif tt_temp[0][0] == 'Topology(Plan)':
            pixmap = QPixmap(self.nl['Topology(Plan)'][0])
            self.fl1.setPixmap(pixmap.scaled(self.fl1.size(),Qt.IgnoreAspectRatio))
            pixmap = QPixmap(self.nl['Topology(Plan)'][1])
            self.fl2.setPixmap(pixmap.scaled(self.fl2.size(),Qt.IgnoreAspectRatio))

        elif tt_temp[0][0] == 'Concept':
            f_a, f_b = text_cal.text_plot(self.nl['Concept'][0], self.nl['Concept'][1])
            self.fl1.setText(f_a)
            self.fl2.setText(f_b)

        elif tt_temp[0][0] == 'Topology(Connection)':
            graph_cal.save_fig_plot(self.nl['Topology(Plan)'][0], self.nl['Topology(Plan)'][1])
            pixmap = QPixmap('./temp_result/tt_i1.png')
            self.fl1.setPixmap(pixmap.scaled(self.fl1.size(),Qt.IgnoreAspectRatio))

            pixmap = QPixmap('./temp_result/tt_i2.png')
            self.fl2.setPixmap(pixmap.scaled(self.fl2.size(),Qt.IgnoreAspectRatio))


        if tt_temp[1][0] == 'Exterior':
            pixmap = QPixmap(self.nl['Exterior'][0])
            self.fl3.setPixmap(pixmap.scaled(self.fl3.size(),Qt.IgnoreAspectRatio))
            pixmap = QPixmap(self.nl['Exterior'][1])
            self.fl4.setPixmap(pixmap.scaled(self.fl4.size(),Qt.IgnoreAspectRatio))

        elif tt_temp[1][0] == 'Interior':
            pixmap = QPixmap(self.nl['Interior'][0])
            self.fl3.setPixmap(pixmap.scaled(self.fl3.size(),Qt.IgnoreAspectRatio))
            pixmap = QPixmap(self.nl['Interior'][1])
            self.fl4.setPixmap(pixmap.scaled(self.fl4.size(),Qt.IgnoreAspectRatio))

        elif tt_temp[1][0] == 'Topology(Plan)':
            pixmap = QPixmap(self.nl['Topology(Plan)'][0])
            self.fl3.setPixmap(pixmap.scaled(self.fl3.size(),Qt.IgnoreAspectRatio))
            pixmap = QPixmap(self.nl['Topology(Plan)'][1])
            self.fl4.setPixmap(pixmap.scaled(self.fl4.size(),Qt.IgnoreAspectRatio))

        elif tt_temp[1][0] == 'Concept':
            f_a, f_b = text_cal.text_plot(self.nl['Concept'][0], self.nl['Concept'][1])
            self.fl3.setText(f_a)
            self.fl4.setText(f_b)

        elif tt_temp[1][0] == 'Topology(Connection)':
            graph_cal.save_fig_plot(self.nl['Topology(Plan)'][0], self.nl['Topology(Plan)'][1])

            pixmap = QPixmap('./temp_result/tt_i1.png')
            self.fl3.setPixmap(pixmap.scaled(self.fl3.size(),Qt.IgnoreAspectRatio))

            pixmap = QPixmap('./temp_result/tt_i2.png')
            self.fl4.setPixmap(pixmap.scaled(self.fl4.size(),Qt.IgnoreAspectRatio))


        self.pbar.setValue(100)


if __name__ == '__main__':
   app = QApplication(sys.argv)
   name_list  = {}
   name_list['Exterior'] = []
   name_list['Interior']= []
   name_list['Topology(Plan)'] = []
   name_list['Concept'] = []
   name_list['Topology(Connection)'] = []
   f_ext = image_cal.f_extract
   ex = MyApp(name_list, f_ext)
   sys.exit(app.exec_())