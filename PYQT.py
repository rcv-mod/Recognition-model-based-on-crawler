    def __init__(self, parent=None):
        super(filedialogdemo, self).__init__(parent)
        layout = QVBoxLayout()

        self.content = QTextEdit()
        layout.addWidget(self.content)

        self.btn = QPushButton()
        self.btn.setText("爬取数据")
        self.btn.clicked.connect(self.buttonClicked)

        self.content2 = QTextEdit()
        layout.addWidget(self.content2)
        self.btn1 = QPushButton()
        self.btn1.setText("训练数据")
        self.btn1.clicked.connect(self.buttonClicked1)

        layout.addWidget(self.btn)
        layout.addWidget(self.btn1)

        self.btn = QPushButton()
        self.btn.clicked.connect(self.loadFile)
        self.btn.setText("从文件中获取照片")
        layout.addWidget(self.btn)

        self.label = QLabel()
        layout.addWidget(self.label)

        self.content1 = QTextEdit()
        layout.addWidget(self.content1)
        self.setWindowTitle("Identity")
        self.setLayout(layout)

    def buttonClicked1(self,keywords):
        self.content2.setPlainText(self.content2.toPlainText())
        string = self.content2.toPlainText()
        file_write_obj = open(r'test.txt', 'a+') # 以写的方式打开文件，如果文件不存在，就会自动创建
        count = len(open(r'test.txt','rU').readlines())
        file_write_obj.writelines(str(count)+":"+string)
        file_write_obj.write('\n')
        file_write_obj.close()
        ff.main()
    def buttonClicked(self,keywords):
        self.content.setPlainText(self.content.toPlainText())
        string = self.content.toPlainText()
        debug.main(string)

    def loadFile(self):
        print("load--file")
        fname, _ = QFileDialog.getOpenFileName(self, '选择图片', r'E:\f\flower_photos', 'Image files(*.jpg *.gif *.png)')
        self.label.setPixmap(QPixmap(fname))
        path = os.getcwd()
        image = Image.open(fname)