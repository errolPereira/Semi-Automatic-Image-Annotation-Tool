import warnings
warnings.filterwarnings('ignore')
#--------------------------- GUI packages _____________________
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from yolo import *
#----------------------- Keras packages -------------------
import keras
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
#--------------------- Keras for resnet model -------------
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image

#--------------- Open Cv packages ------------------------
import cv2
from skimage.util import random_noise

#---------------SSD packages ------------------------------
from keras import backend as K
from keras.optimizers import Adam
from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss

# import miscellaneous modules
import os
import gc
gc.enable()
import numpy as np

#making Tensorflow 2.0 compatible with 1.x
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import config
import math


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

#changing from keras backend to tensorflow backend
tf.keras.backend.set_session(get_session())


######################### models ##############################################

#resnet_model COCO
#path for the saved pretrained model weights
coco_path = os.path.join('.', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')
#loading the model
model_coco = models.load_model(coco_path, backbone_name='resnet50')

#resnet_model Yolo
#path for the saved pretrained model weights
yolo_path = os.path.join('.', 'snapshots', 'yolo.h5')
#loading the model
model_yolo = load_model(yolo_path)

#resnet_model R-CNN
##path for the saved pretrained model weights
#coco_path = os.path.join('.', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')
##loading the model
#model_coco = models.load_model(coco_path, backbone_name='resnet50')
#
#
##resnet_model Fast R-CNN
##path for the saved pretrained model weights
#coco_path = os.path.join('.', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')
##loading the model
#model_coco = models.load_model(coco_path, backbone_name='resnet50')
#
#
##resnet_model Faster R-CNN
##path for the saved pretrained model weights
#coco_path = os.path.join('.', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')
##loading the model
#model_coco = models.load_model(coco_path, backbone_name='resnet50')
#
#
#resnet_model SSD
#path for the saved pretrained model weights
ssd_path = os.path.join('.', 'snapshots', 'VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.h5')
#loading the model
model_ssd = ssd_300(image_size=(300, 300, 3),
                n_classes=20,
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.5,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)
model_ssd.load_weights(ssd_path, by_name=True)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model_ssd.compile(optimizer=adam, loss=ssd_loss.compute_loss)
###############################################################################



#paths of the annotations and classes file
annot_path = 'annotations/annotations.csv'
class_path = 'annotations/classes.csv'

class MainGUI:
    def __init__(self, master):
        
        #default model path
        #path for the saved pretrained model weights
        self.model_path = os.path.join('.', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')
        #loading the model
        self.model = models.load_model(self.model_path, backbone_name='resnet50')

        self.parent = master
        self.center(self.parent, 912, 550)
        self.parent.title("Image Annotator")
        self.frame = Frame(self.parent)
        self.frame.pack(fill=BOTH, expand=1)
        self.parent.resizable(width=False, height=False)

        # Initialize class variables
        self.img = None
        self.img_og = None
        self.tkimg = None
        self.imageDir = ''
        self.imageDirPathBuffer = ''
        self.imageList = []
        self.imageTotal = 0
        self.imageCur = 0
        self.cur = 0
        self.bboxIdList = []
        self.bboxList = []
        self.bboxPointList = []
        self.o1 = None
        self.o2 = None
        self.o3 = None
        self.o4 = None
        self.bboxId = None
        self.currLabel = None
        self.editbboxId = None
        self.currBboxColor = None
        self.zoomImgId = None
        self.zoomImg = None
        self.zoomImgCrop = None
        self.tkZoomImg = None
        self.hl = None
        self.vl = None
        self.editPointId = None
        self.filename = None
        self.filenameBuffer = None
        self.objectLabelList = []
        self.EDIT = False
        self.var = 0.01
        self.mean = 0
        self.amount = 0.5
        self.s_p = 0.5
        self.kernel_size = (4, 4)
        self.type = None
        self.backbone = 'resnet50'
        self.annotation = b'C:\Users\errperei\Desktop\HPE\Errol_docs\ComputerVision\CapgeminiProjects\ImageAnnotation\annotations\annotations.csv'
        self.calsses = b'C:\Users\errperei\Desktop\HPE\Errol_docs\ComputerVision\CapgeminiProjects\ImageAnnotation\annotations\classes.csv'
        self.weights = b'C:\Users\errperei\Desktop\HPE\Errol_docs\ComputerVision\CapgeminiProjects\ImageAnnotation\weights\resnet50_coco_best_v2.1.0.h5'
        self.epochs = 50
        self.steps = 10000
        self.filter_img = None
        self.train_selected = False
        self.class_predict = False

        # initialize mouse state
        self.STATE = {'x': 0, 'y': 0}
        self.STATE_COCO = {'click': 0}

        # initialize annotation file
        self.anno_filename = 'annotations.csv'
        self.annotation_file = open('annotations/' + self.anno_filename, 'w+')
        self.annotation_file.write("")
        self.annotation_file.close()
        
        
        # initialize class file
        self.class_filename = 'classes.csv'
        self.class_file = open('annotations/' + self.class_filename, 'w+')
        self.class_file.write("")
        self.class_file.close()

        # ------------------ GUI ---------------------

        # Control Panel
        self.ctrlPanel = Frame(self.frame)
        self.ctrlPanel.grid(row=0, column=0, sticky=W + N)
        self.openBtn = Button(self.ctrlPanel, text='Open', command=self.open_image)
        self.openBtn.pack(fill=X, side=TOP)
        self.openDirBtn = Button(self.ctrlPanel, text='Open Dir', command=self.open_image_dir)
        self.openDirBtn.pack(fill=X, side=TOP)
        self.nextBtn = Button(self.ctrlPanel, text='Next -->', command=self.open_next)
        self.nextBtn.pack(fill=X, side=TOP)
        self.previousBtn = Button(self.ctrlPanel, text='<-- Previous', command=self.open_previous)
        self.previousBtn.pack(fill=X, side=TOP)
        self.saveBtn = Button(self.ctrlPanel, text='Save', command=self.save)
        self.saveBtn.pack(fill=X, side=TOP)
        self.semiAutoBtn = Button(self.ctrlPanel, text="Show Suggestions", command=self.automate)
        self.semiAutoBtn.pack(fill=X, side=TOP)
        self.disp = Label(self.ctrlPanel, text='Coordinates:')
        self.disp.pack(fill=X, side=TOP)
		
		#Menu button to select the models for detection.
        self.modelMenu = Menubutton(self.ctrlPanel, text="Select Model Default:Resnet", relief=RAISED)
        self.modelMenu.pack(fill=X, side=TOP)
        self.modelMenu.menu = Menu(self.modelMenu, tearoff=0)
        self.modelMenu["menu"] = self.modelMenu.menu 
        
        
        #Menu button to select filters for images.
        self.filterMenu = Menubutton(self.ctrlPanel, text="Select Filters", relief=RAISED)
        self.filterMenu.pack(fill=X, side=TOP)
        self.filterMenu.menu = Menu(self.filterMenu, tearoff=0)
        self.filterMenu["menu"] = self.filterMenu.menu
        
        #Button to add the selected filters
        self.addfilterBtn = Button(self.ctrlPanel, text="+", command=self.add_filter_btn)
        self.addfilterBtn.pack(fill=X, side=TOP)
        self.delfilterBtn = Button(self.ctrlPanel, text="Reset Image", command=self.remove_filter_btn)
        self.delfilterBtn.pack(fill=X, side=TOP)
        
        #MenuButton containing all the labels that COCO model can predict.
        self.mb = Menubutton(self.ctrlPanel, text="Choose Classes", relief=RAISED)
        self.mb.pack(fill=X, side=TOP)
		
#        #Button to add the selected labels
#        self.addCocoBtn = Button(self.ctrlPanel, text="+", command=self.add_labels_coco)
#        self.addCocoBtn.pack(fill=X, side=TOP)
        
        #Zooming panel for the images
        self.zoomPanelLabel = Label(self.ctrlPanel, text="Precision View Panel")
        self.zoomPanelLabel.pack(fill=X, side=TOP)
        self.zoomcanvas = Canvas(self.ctrlPanel, width=150, height=150)
        self.zoomcanvas.pack(fill=X, side=TOP, anchor='center')

        # Image Editing Region
        self.canvas = Canvas(self.frame, width=500, height=500)
        self.canvas.grid(row=0, column=1, sticky=W + N)
        self.canvas.bind("<Button-1>", self.mouse_click)
        self.canvas.bind("<Motion>", self.mouse_move, "+")
        self.canvas.bind("<B1-Motion>", self.mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_release)
        self.parent.bind("<Key-Left>", self.open_previous)
        self.parent.bind("<Key-Right>", self.open_next)
        self.parent.bind("Escape", self.cancel_bbox)

        # Labels and Bounding Box Lists Panel
        self.listPanel = Frame(self.frame)
        self.listPanel.grid(row=0, column=2, sticky=W + N)
        self.listBoxNameLabel = Label(self.listPanel, text="List of Objects").pack(fill=X, side=TOP)
        self.objectListBox = Listbox(self.listPanel, width=40)
        self.objectListBox.pack(fill=X, side=TOP)
        self.delObjectBtn = Button(self.listPanel, text="Delete", command=self.del_bbox)
        self.delObjectBtn.pack(fill=X, side=TOP)
        self.clearAllBtn = Button(self.listPanel, text="Clear All", command=self.clear_bbox)
        self.clearAllBtn.pack(fill=X, side=TOP)
        self.classesNameLabel = Label(self.listPanel, text="Classes").pack(fill=X, side=TOP)
        self.textBox = Entry(self.listPanel, text="Enter label")
        self.textBox.pack(fill=X, side=TOP)
        self.train_model = Button(self.listPanel, text='Train Model', command=self.train_window).pack(fill=X, side=TOP)

        self.addLabelBtn = Button(self.listPanel, text="+", command=self.add_label).pack(fill=X, side=TOP)
        self.delLabelBtn = Button(self.listPanel, text="-", command=self.del_label).pack(fill=X, side=TOP)

        self.labelListBox = Listbox(self.listPanel)
        self.labelListBox.pack(fill=X, side=TOP)
        
        
        #labels for the models
        self.v = IntVar()
        self.v.set(0)        
        self.populate_classes()
        
        ############################ Menu for selecting models ##################
        #Algorithm labels
        self.modelLabels = config.models_to_select.values()
        
        for idxmodel, model_label in enumerate(self.modelLabels):
            self.modelMenu.menu.add_radiobutton(label=model_label, value=idxmodel, variable=self.v, command=self.populate_classes)
        
        ############################# Menu for opencv filters ##################################
        # populating filters
        self.filterIntVars = []
        self.cvfilters = config.opencv_filters.values()
        for idxfilter, filter in enumerate(self.cvfilters):
          self.filterIntVars.append(IntVar())
          self.filterMenu.menu.add_checkbutton(label=filter, variable=self.filterIntVars[idxfilter])

        # STATUS BAR
        self.statusBar = Frame(self.frame, width=500)
        self.statusBar.grid(row=1, column=1, sticky=W + N)
        self.processingLabel = Label(self.statusBar, text="")
        self.processingLabel.pack(side="left", fill=X)
        self.imageIdxLabel = Label(self.statusBar, text="")
        self.imageIdxLabel.pack(side="right", fill=X)
        
        
        #closing
        self.parent.protocol('WM_DELETE_WINDOW', self.on_closing)
    
    def populate_classes(self): 
      algorithm = self.v.get()
      print(algorithm)
      self.cocoIntVars = []
      if (algorithm == 0) or (algorithm == 2):
          print('0 or 2')
          self.labels = config.labels_to_names.values()
      elif algorithm == 1:
          print(1)
          self.labels = config.ssd_classes
      
      print(self.labels)
      

      self.mb.menu = Menu(self.mb, tearoff=0)
      self.mb["menu"] = self.mb.menu
      ########################### Selecting the labels to detect ############################
      for idxcoco, label_coco in enumerate(self.labels):
        print(label_coco)
        self.cocoIntVars.append(IntVar())
        self.mb.menu.add_checkbutton(label=label_coco, variable=self.cocoIntVars[idxcoco])


    def open_image(self):
        self.filename = filedialog.askopenfilename(title="Select Image", filetypes=(("jpeg files", "*.jpg"),
                                                                                    ("all files", "*.*")))
        if not self.filename:
            return None
        self.filenameBuffer = self.filename
        self.load_image(self.filenameBuffer)

    def open_image_dir(self):
        self.imageDir = filedialog.askdirectory(title="Select Dataset Directory")
        if not self.imageDir:
            return None
        self.imageList = os.listdir(self.imageDir)
        self.imageList = sorted(self.imageList)
        self.imageTotal = len(self.imageList)
        self.filename = None
        self.imageDirPathBuffer = self.imageDir
        self.load_image(self.imageDirPathBuffer + '/' + self.imageList[self.cur])

    def load_image(self, file):        
        #opening the image file
        self.img = Image.open(file)
        self.imageCur = self.cur + 1
        self.imageIdxLabel.config(text='  ||   Image Number: %d / %d' % (self.imageCur, self.imageTotal))
        
        # Resize to Pascal VOC format
        w, h = self.img.size
        if w >= h:
            baseW = 500
            wpercent = (baseW / float(w))
            hsize = int((float(h) * float(wpercent)))
            self.img = self.img.resize((baseW, hsize), Image.BICUBIC)
        else:
            baseH = 500
            wpercent = (baseH / float(h))
            wsize = int((float(w) * float(wpercent)))
            self.img = self.img.resize((wsize, baseH), Image.BICUBIC)

        self.display_img(self.img)
        self.img_og = self.img
        self.automate()
        
    def open_next(self, event=None):
        self.save_filtered_img()
        self.save()
        if self.cur < len(self.imageList):
            self.cur += 1
            self.load_image(self.imageDirPathBuffer + '/' + self.imageList[self.cur])
        self.processingLabel.config(text="")
        self.processingLabel.update_idletasks()

    def save_filtered_img(self):
        if self.filter_img:
            self.filter_img.save(self.imageDir + '/filter_' +self.imageList[self.cur]) 
            self.imageList[self.cur] = 'filter_' + self.imageList[self.cur]
            self.filter_img = None
            self.processingLabel.config(text="")
            self.processingLabel.update_idletasks()
        else:
            pass
    
    def open_previous(self, event=None):
        self.save_filtered_img()
        self.save()
        if self.cur > 0:
            self.cur -= 1
            self.load_image(self.imageDirPathBuffer + '/' + self.imageList[self.cur])
        self.processingLabel.config(text="                      ")
        self.processingLabel.update_idletasks()

    def save(self):
        if self.filenameBuffer is None:
            self.annotation_file = open('annotations/' + self.anno_filename, 'a')
            for idx, item in enumerate(self.bboxList):
                self.annotation_file.write(self.imageDirPathBuffer + '/' + self.imageList[self.cur] + ',' +
                                           ','.join(map(str, self.bboxList[idx])) + ',' + str(self.objectLabelList[idx])
                                           + '\n')
            self.annotation_file.close()
        else:
            self.annotation_file = open('annotations/' + self.anno_filename, 'a')
            for idx, item in enumerate(self.bboxList):
                self.annotation_file.write(self.filenameBuffer + ',' + ','.join(map(str, self.bboxList[idx])) + ','
                                           + str(self.objectLabelList[idx]) + '\n')
            self.annotation_file.close()
        self.populate_listbox()
    
    def populate_listbox(self):
        curr_label_list = []
        for x in self.objectLabelList:
            curr_label_list.append(x)
        curr_label_list = list(set(curr_label_list))
        labelList = self.labelListBox.get(0, END)
        labelList = list(labelList)
        for item in curr_label_list:
            if item not in labelList:
                self.labelListBox.insert(END, str(item))

    def mouse_click(self, event):
        # Check if Updating BBox
        if self.canvas.find_enclosed(event.x - 5, event.y - 5, event.x + 5, event.y + 5):
            self.EDIT = True
            self.editPointId = int(self.canvas.find_enclosed(event.x - 5, event.y - 5, event.x + 5, event.y + 5)[0])
        else:
            self.EDIT = False

        # Set the initial point
        if self.EDIT:
            idx = self.bboxPointList.index(self.editPointId)
            self.editbboxId = self.bboxIdList[math.floor(idx/4.0)]
            self.bboxId = self.editbboxId
            pidx = self.bboxIdList.index(self.editbboxId)
            pidx = pidx * 4
            self.o1 = self.bboxPointList[pidx]
            self.o2 = self.bboxPointList[pidx + 1]
            self.o3 = self.bboxPointList[pidx + 2]
            self.o4 = self.bboxPointList[pidx + 3]
            if self.editPointId == self.o1:
                a, b, c, d = self.canvas.coords(self.o3)
            elif self.editPointId == self.o2:
                a, b, c, d = self.canvas.coords(self.o4)
            elif self.editPointId == self.o3:
                a, b, c, d = self.canvas.coords(self.o1)
            elif self.editPointId == self.o4:
                a, b, c, d = self.canvas.coords(self.o2)
            self.STATE['x'], self.STATE['y'] = int((a+c)/2), int((b+d)/2)
        else:
            self.STATE['x'], self.STATE['y'] = event.x, event.y

    def mouse_drag(self, event):
        self.mouse_move(event)
        if self.bboxId:
            self.currBboxColor = self.canvas.itemcget(self.bboxId, "outline")
            self.canvas.delete(self.bboxId)
            self.canvas.delete(self.o1)
            self.canvas.delete(self.o2)
            self.canvas.delete(self.o3)
            self.canvas.delete(self.o4)
        if self.EDIT:
            self.bboxId = self.canvas.create_rectangle(self.STATE['x'], self.STATE['y'],
                                                       event.x, event.y,
                                                       width=1,
                                                       outline=self.currBboxColor)
        else:
            self.currBboxColor = config.COLORS[len(self.bboxList) % len(config.COLORS)]
            self.bboxId = self.canvas.create_rectangle(self.STATE['x'], self.STATE['y'],
                                                       event.x, event.y,
                                                       width=1,
                                                       outline=self.currBboxColor)

    def mouse_move(self, event):
        self.disp.config(text='x: %d, y: %d' % (event.x, event.y))
        self.zoom_view(event)
        if self.tkimg:
            # Horizontal and Vertical Line for precision
            if self.hl:
                self.canvas.delete(self.hl)
            self.hl = self.canvas.create_line(0, event.y, self.tkimg.width(), event.y, width=1)
            if self.vl:
                self.canvas.delete(self.vl)
            self.vl = self.canvas.create_line(event.x, 0, event.x, self.tkimg.height(), width=1)
            # elif (event.x, event.y) in self.bboxBRPointList:
            #     pass

    def mouse_release(self, event):
        try:
            labelidx = self.labelListBox.curselection()
            self.currLabel = self.labelListBox.get(labelidx)
        except:
            pass
        if self.EDIT:
            self.update_bbox()
            self.EDIT = False
        x1, x2 = min(self.STATE['x'], event.x), max(self.STATE['x'], event.x)
        y1, y2 = min(self.STATE['y'], event.y), max(self.STATE['y'], event.y)
        self.bboxList.append((x1, y1, x2, y2))
        o1 = self.canvas.create_oval(x1 - 3, y1 - 3, x1 + 3, y1 + 3, fill="red")
        o2 = self.canvas.create_oval(x2 - 3, y1 - 3, x2 + 3, y1 + 3, fill="red")
        o3 = self.canvas.create_oval(x2 - 3, y2 - 3, x2 + 3, y2 + 3, fill="red")
        o4 = self.canvas.create_oval(x1 - 3, y2 - 3, x1 + 3, y2 + 3, fill="red")
        self.bboxPointList.append(o1)
        self.bboxPointList.append(o2)
        self.bboxPointList.append(o3)
        self.bboxPointList.append(o4)
        self.bboxIdList.append(self.bboxId)
        self.bboxId = None
        self.objectLabelList.append(str(self.currLabel))
        self.objectListBox.insert(END, '(%d, %d) -> (%d, %d)' % (x1, y1, x2, y2) + ': ' + str(self.currLabel))
        self.objectListBox.itemconfig(len(self.bboxIdList) - 1,
                                      fg=self.currBboxColor)
        self.currLabel = None

    def zoom_view(self, event):
        try:
            if self.zoomImgId:
                self.zoomcanvas.delete(self.zoomImgId)
            self.zoomImg = self.img.copy()
            self.zoomImgCrop = self.zoomImg.crop(((event.x - 25), (event.y - 25), (event.x + 25), (event.y + 25)))
            self.zoomImgCrop = self.zoomImgCrop.resize((150, 150))
            self.tkZoomImg = ImageTk.PhotoImage(self.zoomImgCrop)
            self.zoomImgId = self.zoomcanvas.create_image(0, 0, image=self.tkZoomImg, anchor=NW)
            hl = self.zoomcanvas.create_line(0, 75, 150, 75, width=2)
            vl = self.zoomcanvas.create_line(75, 0, 75, 150, width=2)
        except:
            pass

    def update_bbox(self):
        idx = self.bboxIdList.index(self.editbboxId)
        self.bboxIdList.pop(idx)
        self.bboxList.pop(idx)
        self.objectListBox.delete(idx)
        self.currLabel = self.objectLabelList[idx]
        self.objectLabelList.pop(idx)
        idx = idx*4
        self.canvas.delete(self.bboxPointList[idx])
        self.canvas.delete(self.bboxPointList[idx+1])
        self.canvas.delete(self.bboxPointList[idx+2])
        self.canvas.delete(self.bboxPointList[idx+3])
        self.bboxPointList.pop(idx)
        self.bboxPointList.pop(idx)
        self.bboxPointList.pop(idx)
        self.bboxPointList.pop(idx)

    def cancel_bbox(self, event):
        if self.STATE['click'] == 1:
            if self.bboxId:
                self.canvas.delete(self.bboxId)
                self.bboxId = None
                self.STATE['click'] = 0

    def del_bbox(self):
        sel = self.objectListBox.curselection()
        if len(sel) != 1:
            return
        idx = int(sel[0])
        self.canvas.delete(self.bboxIdList[idx])
        self.canvas.delete(self.bboxPointList[idx * 4])
        self.canvas.delete(self.bboxPointList[(idx * 4) + 1])
        self.canvas.delete(self.bboxPointList[(idx * 4) + 2])
        self.canvas.delete(self.bboxPointList[(idx * 4) + 3])
        self.bboxPointList.pop(idx * 4)
        self.bboxPointList.pop(idx * 4)
        self.bboxPointList.pop(idx * 4)
        self.bboxPointList.pop(idx * 4)
        self.bboxIdList.pop(idx)
        self.bboxList.pop(idx)
        self.objectLabelList.pop(idx)
        self.objectListBox.delete(idx)

    def clear_bbox(self):
        for idx in range(len(self.bboxIdList)):
            self.canvas.delete(self.bboxIdList[idx])
        for idx in range(len(self.bboxPointList)):
            self.canvas.delete(self.bboxPointList[idx])
        self.objectListBox.delete(0, len(self.bboxList))
        self.bboxIdList = []
        self.bboxList = []
        self.objectLabelList = []
        self.bboxPointList = []

    def add_label(self):
        if self.textBox.get() is not '':
            curr_label_list = self.labelListBox.get(0, END)
            curr_label_list = list(curr_label_list)
            if self.textBox.get() not in curr_label_list:
                self.labelListBox.insert(END, str(self.textBox.get()))
            self.textBox.delete(0, 'end')

    def del_label(self):
        labelidx = self.labelListBox.curselection()
        self.labelListBox.delete(labelidx)

#    def add_labels_coco(self):
#        for listidxcoco, list_label_coco in enumerate(self.cocoLabels):
#            if self.cocoIntVars[listidxcoco].get():
#                curr_label_list = self.labelListBox.get(0, END)
#                curr_label_list = list(curr_label_list)
#                if list_label_coco not in curr_label_list:
#                    self.labelListBox.insert(END, str(list_label_coco))
                    
    def add_filter_btn(self):
        curr_filter = []
        for filteridx, filter in enumerate(self.cvfilters):
            if self.filterIntVars[filteridx].get():
                if curr_filter not in [filter]:
                    curr_filter.append(filter)
        
        if curr_filter == []:
            self.display_img(self.img_og)
            self.img = self.img_og
            pass
          
        #getting the image
        for x in curr_filter:
            if x == 'Blur':
                self.pop_param('blur')
                self.parent.wait_window(self.win)
                continue
            if x == 'Gaussian Noise':
                self.pop_param('gauss')
                self.parent.wait_window(self.win)
                continue
            if x == 'S&P':
                self.pop_param('s&p')
                self.parent.wait_window(self.win)
                continue
            if x == 'Poisson':
                #calling the noise function
                self.img = self.noisy('poisson', np.array(self.img))
                print(self.img.shape, type(self.img))
                #converting the numpy base array to an PIL image format
                self.img = Image.fromarray((self.img * 255).astype(np.uint8))
                #display image on canvas
                self.display_img(self.img)
                
            if x == 'Speckle':
                self.pop_param('speck')
                self.parent.wait_window(self.win)
                continue
    

    #-------------------function to add noise in the image -------------------------
    
    def noisy(self, noise_typ, image):
       if noise_typ == "gauss":
          noisy_image = random_noise(image,
                                     mode='gaussian',
                                     seed=None,
                                     clip=True,
                                     mean=self.mean,
                                     var=self.var)
          return noisy_image
       elif noise_typ == "s&p":
         noisy_image = random_noise(image,
                                    mode='s&p',
                                    seed=None,
                                    clip=True,
                                    amount=self.amount,
                                    salt_vs_pepper=self.s_p)
         return noisy_image
         
       elif noise_typ == "poisson":
          noisy_image = random_noise(image,
                                     mode='poisson',
                                     seed=None,
                                     clip=True)
          return noisy_image
       elif noise_typ =="speckle":
          noisy_image = random_noise(image,
                                     mode='speckle',
                                     seed=None,
                                     clip=True,
                                     mean=self.mean,
                                     var=self.var)
          return noisy_image

    #---------------------funtion to get parameters of filters from user------------------
    def pop_param(self, noise_typ):
      self.win = Toplevel(width=460, height=350)
      self.center(self.win, 380, 100)
      
      #------------------------Gaussian Noise---------------------------------
      if noise_typ == 'gauss' or noise_typ == 'speck':
         self.type = noise_typ
         self.win.wm_title(f'{noise_typ} Parameters') 
         
         #heading for the form
         heading = Label(self.win, text=f'Select {noise_typ} parameters')
         #parameter labels
         mean = Label(self.win, text='Mean')
         var = Label(self.win, text='Variance')
         
         #params placement
         heading.grid(row=0, column=1)
         mean.grid(row=1, column=0)
         var.grid(row=2, column=0)
         
         #entry box/input box
         self.mean_field = Entry(self.win)
         self.mean_field.grid(row=1, column=1, ipadx='80')

         
         self.var_field = Entry(self.win)
         self.var_field.grid(row=2, column=1, ipadx='80')
         
         submit = Button(self.win, text="Submit", fg="Black", 
                            bg="Red", command=self.set_input)
         submit.grid(row=3, column=1)
         
      #--------------------------------blur------------------------------------
      if noise_typ == 'blur':
        self.type = 'blur'
        self.win.wm_title('Blur Kernel Size')
        
        #heading for the form
        heading = Label(self.win, text='Select Kenel size (4, 4)')
        #parameter labels
        kernel = Label(self.win, text='Kernel Size')
        
        #params placement
        heading.grid(row=0, column=1)
        kernel.grid(row=1, column=0)
        
        #entry box/input box
        self.kernel_field = Entry(self.win)
        self.kernel_field.grid(row=1, column=1, ipadx='40')
        
        submit = Button(self.win, text="Submit", fg="Black", 
                            bg="Red", command=self.set_input)
        submit.grid(row=2, column=1)
        
      if noise_typ == 's&p':
        self.type = 's&p'
        self.win.wm_title('Salt & Pepper params')
        
        #heading for the form
        heading = Label(self.win, text='Select S&P parameters')
        #parameter labels
        amount = Label(self.win, text='Amount (range [0, 1])')
        svp = Label(self.win, text='Salt vs Pepper proportion (range [0, 1])')
        
        #params placement
        heading.grid(row=0, column=1)
        amount.grid(row=1, column=0)
        svp.grid(row=2, column=0)
        
        #entry box/input box
        self.amount_field = Entry(self.win)
        self.amount_field.grid(row=1, column=1, ipadx='20')
        self.svp_field = Entry(self.win)
        self.svp_field.grid(row=2 , column=1, ipadx='20')
        
        submit = Button(self.win, text="Submit", fg="Black", 
                            bg="Red", command=self.set_input)
        submit.grid(row=3, column=1)
        
         
    #function to set the input
    def set_input(self):
        if self.type == 'gauss':
          self.mean = np.float32(self.mean_field.get())
          self.var = np.float32(self.var_field.get())
          #calling the noise function
          self.img = self.noisy('gauss', np.array(self.img))
          #converting the numpy base array to an PIL image format
          self.img = Image.fromarray((self.img * 255).astype(np.uint8))
          self.filter_img = self.img
          #display image on canvas
          self.display_img(self.img)
          self.win.destroy()
          
        if self.type == 'speck':
          self.mean = np.float32(self.mean_field.get())
          self.var = np.float32(self.var_field.get())
          #calling the noise function
          self.img = self.noisy('speckle', np.array(self.img))
          #converting the numpy base array to an PIL image format
          self.img = Image.fromarray((self.img * 255).astype(np.uint8))
          self.filter_img = self.img
          #display image on canvas
          self.display_img(self.img)
          self.win.destroy()
        
        if self.type == 'blur':
          k_sz = [int(x) for x in self.kernel_field.get().split(',')]
          self.kernel_size = tuple(k_sz)
           #applying blur to the image with kernel size (4, 4)
          self.img = cv2.blur(np.array(self.img), self.kernel_size)
          #converting the numpy base array to an PIL image format
          self.img = Image.fromarray(self.img)
          self.filter_img = self.img
          #display image on canvas
          self.display_img(self.img)
          self.win.destroy()
        
        if self.type == 's&p':
          self.amount = np.float32(self.amount_field.get())
          self.s_p = np.float32(self.svp_field.get())
          #calling the noise function
          self.img = self.noisy('s&p', np.array(self.img))
          #converting the numpy base array to an PIL image format
          self.img = Image.fromarray((self.img * 255).astype(np.uint8))
          self.filter_img = self.img
          #display image on canvas
          self.display_img(self.img)
          self.win.destroy()
        
    #function to display the image
    def display_img(self, image):
        #code to display the image on the canvas
        self.tkimg = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, image=self.tkimg, anchor=NW)
        self.clear_bbox()
    
    def remove_filter_btn(self):
      self.display_img(self.img_og)
      self.img = self.img_og

    def automate(self):
        open_cv_image = np.array(self.img)
        # Convert RGB to BGR
        opencvImage= open_cv_image[:, :, ::-1].copy()
        # opencvImage = cv2.cvtColor(np.array(self.img), cv2.COLOR_RGB2BGR)
        algorithm = self.v.get()
        temp = []
        listcoco = []
        for listidxcoco, list_label_coco in enumerate(self.labels):
            temp.append(self.cocoIntVars[listidxcoco].get())
            if self.cocoIntVars[listidxcoco].get():
                listcoco.append(list_label_coco)
        
        if sum(temp):
            self.class_predict = True
        else:
            self.class_predict = False

        ################################### COCO Model ##################################
        if algorithm == 0:
            self.processingLabel.config(text="Processing")
            self.processingLabel.update_idletasks()
            image = preprocess_image(opencvImage)
            boxes, scores, labels = model_coco.predict_on_batch(np.expand_dims(image, axis=0))
            for idx, (box, label, score) in enumerate(zip(boxes[0], labels[0], scores[0])):
                if score < 0.5:
                    continue
                
                if self.class_predict:    
                    if config.labels_to_names[label] not in listcoco:
                        continue
    
                b = box.astype(int)
    
                self.bboxId = self.canvas.create_rectangle(b[0], b[1],
                                                           b[2], b[3],
                                                           width=1,
                                                           outline=config.COLORS[len(self.bboxList) % len(config.COLORS)])
                self.bboxList.append((b[0], b[1], b[2], b[3]))
                o1 = self.canvas.create_oval(b[0] - 3, b[1] - 3, b[0] + 3, b[1] + 3, fill="red")
                o2 = self.canvas.create_oval(b[2] - 3, b[1] - 3, b[2] + 3, b[1] + 3, fill="red")
                o3 = self.canvas.create_oval(b[2] - 3, b[3] - 3, b[2] + 3, b[3] + 3, fill="red")
                o4 = self.canvas.create_oval(b[0] - 3, b[3] - 3, b[0] + 3, b[3] + 3, fill="red")
                self.bboxPointList.append(o1)
                self.bboxPointList.append(o2)
                self.bboxPointList.append(o3)
                self.bboxPointList.append(o4)
                self.bboxIdList.append(self.bboxId)
                self.bboxId = None
                self.objectLabelList.append(str(config.labels_to_names[label]))
                self.objectListBox.insert(END, '(%d, %d) -> (%d, %d)' % (b[0], b[1], b[2], b[3]) + ': ' +
                                          str(config.labels_to_names[label]))
                self.objectListBox.itemconfig(len(self.bboxIdList) - 1,
                                              fg=config.COLORS[(len(self.bboxIdList) - 1) % len(config.COLORS)])
            self.processingLabel.config(text="Done")
            
            
        ################################ SSD Model #####################################
        if algorithm == 1:
            self.processingLabel.config(text="Processing")
            self.processingLabel.update_idletasks()
            width, height = self.img.size
            image = self.img
            image = image.resize((300, 300), Image.BICUBIC)
            image = img_to_array(image)
            # add a dimension so that we have one sample
            image = expand_dims(image, 0)
            
            y_pred = model_ssd.predict(image)
            
            classes = config.ssd_classes
            
            confidence_threshold = 0.7
            
            y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

            np.set_printoptions(precision=2, suppress=True, linewidth=90)
            

            for box in y_pred_thresh[0]:
                label = classes[int(box[0])]
                if self.class_predict:
                    if label not in listcoco:
                        continue
                # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
                x1 = int(box[2] * width / 300)
                y1 = int(box[3] * height / 300)
                x2 = int(box[4] * width / 300)
                y2 = int(box[5] * height / 300)
                

                
                self.bboxId = self.canvas.create_rectangle(x1, y1, 
                                                           x2, y2,
                                                           width=1,
                                                           outline=config.COLORS[len(self.bboxList) % len(config.COLORS)])
                self.bboxList.append((x1, y1, x2, y2))
                o1 = self.canvas.create_oval(x1 - 3, y1 - 3, x1 + 3, y1 + 3, fill="red")
                o2 = self.canvas.create_oval(x2 - 3, y1 - 3, x2 + 3, y1 + 3, fill="red")
                o3 = self.canvas.create_oval(x2 - 3, y2 - 3, x2 + 3, y2 + 3, fill="red")
                o4 = self.canvas.create_oval(x1 - 3, y2 - 3, x1 + 3, y2 + 3, fill="red")
                self.bboxPointList.append(o1)
                self.bboxPointList.append(o2)
                self.bboxPointList.append(o3)
                self.bboxPointList.append(o4)
                self.bboxIdList.append(self.bboxId)
                self.bboxId = None
                self.objectLabelList.append(str(label))
                self.objectListBox.insert(END, '(%d, %d) -> (%d, %d)' % (x1, y1, x2, y2) + ': ' +
                                          str(label))
                self.objectListBox.itemconfig(len(self.bboxIdList) - 1,
                                              fg=config.COLORS[(len(self.bboxIdList) - 1) % len(config.COLORS)])
            self.processingLabel.config(text="Done")
        
        ################################# YOLO MODEL ###################################
        if algorithm == 2:
            self.processingLabel.config(text="Processing")
            self.processingLabel.update_idletasks()
            width, height = self.img.size
            image = self.img
            image = image.resize((416, 416), Image.BICUBIC)
            image = img_to_array(image)
            # scale pixel values to [0, 1]
            image = image.astype('float32')
            image /= 255.0
            # add a dimension so that we have one sample
            image = expand_dims(image, 0)
            yhat = model_yolo.predict(image)
            
            
            anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
            # define the probability threshold for detected objects
            class_threshold = 0.6
            boxes = list()
            for i in range(len(yhat)):
                # decode the output of the network
                boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, 416, 416)
            # correct the sizes of the bounding boxes for the shape of the image
            correct_yolo_boxes(boxes, height, width, 416, 416)
            # suppress non-maximal boxes
            do_nms(boxes, 0.5)
            labels = list(config.labels_to_names.values())
            # get the details of the detected objects
            v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)
            # summarize what we found
            for i in range(len(v_boxes)):
                box = v_boxes[i]
                # get coordinates
                y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
                self.bboxId = self.canvas.create_rectangle(x1, y1, 
                                                           x2, y2,
                                                           width=1,
                                                           outline=config.COLORS[len(self.bboxList) % len(config.COLORS)])
                self.bboxList.append((int(x1), int(y1), int(x2), int(y2)))
                o1 = self.canvas.create_oval(x1 - 3, y1 - 3, x1 + 3, y1 + 3, fill="red")
                o2 = self.canvas.create_oval(x2 - 3, y1 - 3, x2 + 3, y1 + 3, fill="red")
                o3 = self.canvas.create_oval(x2 - 3, y2 - 3, x2 + 3, y2 + 3, fill="red")
                o4 = self.canvas.create_oval(x1 - 3, y2 - 3, x1 + 3, y2 + 3, fill="red")
                self.bboxPointList.append(o1)
                self.bboxPointList.append(o2)
                self.bboxPointList.append(o3)
                self.bboxPointList.append(o4)
                self.bboxIdList.append(self.bboxId)
                self.bboxId = None
                self.objectLabelList.append(str(v_labels[i]))
                self.objectListBox.insert(END, '(%d, %d) -> (%d, %d)' % (x1, y1, x2, y2) + ': ' +
                                          str(v_labels[i]))
                self.objectListBox.itemconfig(len(self.bboxIdList) - 1,
                                              fg=config.COLORS[(len(self.bboxIdList) - 1) % len(config.COLORS)])
            self.processingLabel.config(text="Done")
    
    #function for displaying the training window      
    def train_window(self): 
        self.save_classes()
        self.train_win = Toplevel()
        self.center(self.train_win, 360, 200)
        
        self.train_win.wm_title(f'Training Parameters')
        self.train_win.iconphoto(False, imgicon)
        
        #keeping win in focus
        self.train_win.focus_force() 
        vcmd = (self.parent.register(self.validateTrain),
                '%d', '%S')
        
        heading = Label(self.train_win, text='Please enter the below parameters')
        backbone = Label(self.train_win, text='Backbone')
        epochs = Label(self.train_win, text='Epochs')
        steps = Label(self.train_win, text='Steps Size')
        annotation = Label(self.train_win, text='Annotations file path')
        classes = Label(self.train_win, text='Classs file path')
        weights = Label(self.train_win, text='Weights file path')
        
        heading.grid(row=0, column=1)
        backbone.grid(row=1, column=0)
        epochs.grid(row=2, column=0)
        steps.grid(row=3, column=0)
        annotation.grid(row=4, column=0)
        classes.grid(row=5, column=0)
        weights.grid(row=6, column=0)

      
        self.backbone_field = Entry(self.train_win)
        self.epochs_field = Entry(self.train_win, validate="key", validatecommand=vcmd)
        self.text = Label(self.train_win, text='')
        self.steps_field = Entry(self.train_win)
        self.annotation_field = Button(self.train_win, text='Select file', command=self.select_annot_file)
        self.classes_field = Button(self.train_win, text='Select file', command=self.select_class_file)                                                                                  
        self.weights_field = Button(self.train_win, text='Select file', command=self.select_h5_file)
        
        self.backbone_field.grid(row=1, column=1, ipadx='50')
        self.epochs_field.grid(row=2, column=1, ipadx='50')
        self.text.grid(row=2, column=2)
        self.steps_field.grid(row=3, column=1, ipadx='50')
        self.annotation_field.grid(row=4, column=1, ipadx='50')
        self.classes_field.grid(row=5, column=1, ipadx='50')
        self.weights_field.grid(row=6, column=1, ipadx='50')

        submit = Button(self.train_win, text="Start Training", fg="Black", 
                            bg="blue", command=self.training_model)
        submit.grid(row=7, column=1)
        
    
    #function for validating training input 
    def validateTrain(self, d, S):
        self.text.config(text='')
        self.text.config(text='*Only Numbers', fg='red')
        # Disallow anything but numbers
        if S.isdigit():
            self.text.config(text='')
            return True
        else:
            self.train_win.bell()
            return False
    

    #functions for selecting the training files
    #-----------------------------------------------------------------------------------------------------    
    def select_annot_file(self):
        self.annotation = filedialog.askopenfilename(parent=self.train_win, title="Select File", filetypes=(("csv files", "*.csv"),
                                                                                    ("all files", "*.*")), initialdir='annotations/')
    
    def select_class_file(self):
        self.classes = filedialog.askopenfilename(parent=self.train_win, title="Select File", filetypes=(("csv files", "*.csv"),
                                                                                    ("all files", "*.*")), initialdir='annotations/')
        
    def select_h5_file(self):
        self.weights = filedialog.askopenfilename(parent=self.train_win, title="Select File", filetypes=(("csv files", "*.h5"),
                                                                                   ("all files", "*.*")), initialdir='weights/')
    #-----------------------------------------------------------------------------------------------------

    
    def training_model(self):
        if self.backbone_field.get():
            self.backbone = self.backbone_field.get()
        if self.epochs_field.get():
            self.epochs = self.epochs_field.get()  
        if self.steps_field.get():
            self.steps = self.steps_field.get() 
            
        command = f'retinanet-train --backbone {self.backbone} --epochs {int(self.epochs)} --steps {self.steps} --weights {self.weights} csv {self.annotation} {self.classes} '
        print(command)
        gc.collect()
        return_status = os.system(command)
        if return_status == 0:
            messagebox.showinfo('Success', 'Training completed successfully!')
        else:
            messagebox.showinfo('Error', 'Errors encountered while training')
        self.train_win.destroy()
        
    def save_classes(self):
        listbox = self.labelListBox.get(0, END)
        listbox = list(listbox)
        
        self.class_file = open('annotations/' + self.class_filename, 'w+')
        for idx, item in enumerate(listbox):
            self.class_file.write(item + ',' + str(idx) + '\n')
        self.class_file.close()
            
    def on_closing(self):    
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.parent.destroy()
                
    def center(self, root, width, height):
        if (width == None) & (height == None):
            print('Enter')
            print(root.winfo_width())
            print(root.winfo_height())
            width = root.winfo_width()
            height = root.winfo_height()
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry('{}x{}+{}+{}'.format(width, height, x, y)) 
        
if __name__ == '__main__':
    root = Tk()
    imgicon = PhotoImage(file='icon.png')
    root.iconphoto(False,  imgicon)
    tool = MainGUI(root)  
    gc.collect()
    root.mainloop()