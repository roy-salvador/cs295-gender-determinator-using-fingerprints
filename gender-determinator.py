import Tkinter, Tkconstants, tkFileDialog
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import caffe
from decimal import Decimal

# Load and Initialize Models
FEMALE = 0
MALE = 1
lenet = caffe.Net('model/lenet_deploy.prototxt', 'model/SAVE/lenet_iter_10000_1.caffemodel', caffe.TEST)
alexnet = caffe.Net('model/alexnet_deploy.prototxt', 'model/SAVE/alexnet_iter_10000_3.caffemodel', caffe.TEST)
maxout = caffe.Net('model/alexnet-maxout_deploy.prototxt', 'model/SAVE/alexnet-maxout_iter_10000_1.caffemodel', caffe.TEST)
elu = caffe.Net('model/elu_deploy.prototxt', 'model/SAVE/elu_iter_10000_1.caffemodel', caffe.TEST)
elu_oversampled = caffe.Classifier('model/elu_oversampled_deploy.prototxt','model/SAVE/elu_iter_10000_augmented.caffemodel', image_dims=(256,256))


class GenderDeterminatorFrame(Tkinter.Frame):

  # Initializes the Main Application Frame
  def __init__(self, root):
    Tkinter.Frame.__init__(self, root)
	
	# default_mode
    self.root = root
    self.state = 'normal'
	
	# Layout 
    Tkinter.Label(self, text="Current Fingerprint", font=("Helvetica", 12)).grid(row=0, column=2, columnspan = 3)
    self.currentFile = Tkinter.Label(self, text="None", font=("Helvetica", 12), fg="darkgreen")
    self.currentFile.grid(row=1, column=2, columnspan = 3)
    Tkinter.Label(self, text="             ", font=("Helvetica", 12)).grid(row=3, column=1)
    self.button_load = Tkinter.Button(self, text='Load File', command=self.loadFingerprint)
    Tkinter.Label(self, text="             ", font=("Helvetica", 12)).grid(row=4, column=1)
    Tkinter.Label(self, text="   ", font=("Helvetica", 12)).grid(row=4, column=2)
    self.button_load.grid(padx=5, pady=10, row=3, column=2, columnspan=3)
    Tkinter.Label(self, text="         Groundtruth:", font=("Helvetica", 12)).grid(row=5, column=2)
    Tkinter.Label(self, text="               LeNet:", font=("Helvetica", 12)).grid(row=6, column=2)
    Tkinter.Label(self, text="             AlexNet:", font=("Helvetica", 12)).grid(row=7, column=2)
    Tkinter.Label(self, text="              Maxout:", font=("Helvetica", 12)).grid(row=8, column=2)
    Tkinter.Label(self, text="                 ELU:", font=("Helvetica", 12)).grid(row=9, column=2)
    Tkinter.Label(self, text=" ELU w/ Oversampling:", font=("Helvetica", 12)).grid(row=10, column=2)	
    self.groundtruth = Tkinter.Label(self, text="                ", font=("Helvetica", 12))
    self.groundtruth.grid(row=5, column=4)
    self.lenetResult = Tkinter.Label(self, text="                ", font=("Helvetica", 12))
    self.lenetResult.grid(row=6, column=4)
    self.alexnetResult = Tkinter.Label(self, text="                ", font=("Helvetica", 12))
    self.alexnetResult.grid(row=7, column=4)
    self.maxoutResult = Tkinter.Label(self, text="                ", font=("Helvetica", 12))
    self.maxoutResult.grid(row=8, column=4)
    self.eluResult = Tkinter.Label(self, text="                ", font=("Helvetica", 12))
    self.eluResult.grid(row=9, column=4)
    self.eluOversampledResult = Tkinter.Label(self, text="                ", font=("Helvetica", 12))
    self.eluOversampledResult.grid(row=10, column=4)
	
	# define options for opening or saving a file
    self.file_opt = options = {}
    options['defaultextension'] = '.png'
    options['filetypes'] = [('png files', '.png'), ('jpg files', '.jpg')]
    options['parent'] = root
    options['title'] = 'Load Fingerprint file'
	
  # Predict gender based on image
  def predictGender(self, net, image, label) :
    
    
    if label != self.eluOversampledResult :
      # Configure / preprocessing
      transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
      transformer.set_transpose('data', (2,0,1))
      net.blobs['data'].data[...] = transformer.preprocess('data', image)
      net.blobs['data'].reshape(1,1,256,256)
        
      #compute
      out = net.forward()
      predictedClass = out['prob'].argmax()
      
      resultString = ''
      if predictedClass == FEMALE :
        resultString = 'Female ('
        label['fg'] = 'red'
      else :
        resultString = 'Male ('
        label['fg'] = 'blue'
        
      confidence = round(Decimal(out['prob'][0][predictedClass] * 100.0), 2)
      resultString = resultString + str(confidence) + '%)'
      label['text'] = resultString
    
    # For ensemble of ELU Oversampled
    else :
      out = net.predict(np.array([image]))
      predictedClass = out.argmax()
      
      femaleProb = out[0][0]
      maleProb = out[0][1]
      predictedClass = 1
      if femaleProb >= maleProb :
        predictedClass = 0
        label['fg'] = 'red'
        #resultString = 'Female'
        resultString = 'Female (' + str(round(Decimal(femaleProb * 100.0 / (femaleProb + maleProb)), 2))  + '%)'
      else :
        label['fg'] = 'blue'
        #resultString = 'Male'
        resultString = 'Male (' + str(round(Decimal(maleProb * 100.0 / (femaleProb + maleProb)) , 2))  + '%)'
      label['text'] = resultString
      
      
        
      
   	

  # Displays Fingerprint Image 
  def displayFingerprint(self, filename) :
    img = cv2.imread(filename,0)
    cv2.imshow(os.path.basename(filename) ,img)
    #plt.close()
    #plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    #plt.show()
    #plt.close()
    
	

  # Loads a saved fingerprint file
  def loadFingerprint(self):
    # get filename
    filename = tkFileDialog.askopenfilename(**self.file_opt)

    # open and display file
    if filename:
      self.currentFile['text'] = os.path.basename(filename) 
      
      # Display ground truth if present
      groundtruthFilename = filename.replace(".png", ".txt").replace(".jpg", ".txt")
      if os.path.isfile(groundtruthFilename) :
        labelsFile = open(groundtruthFilename, 'r')
        gender = ''
        for line in labelsFile :
            gender = line[len(line) - 2]
            break
			
        if gender == 'M' :
          self.groundtruth['text'] = "Male"
          self.groundtruth['fg'] = "blue"
        else :
          self.groundtruth['text'] = "Female"
          self.groundtruth['fg'] = "red"
            
      else :
        self.groundtruth['text'] = "Unknown"
        self.groundtruth['fg'] = "black"
      	  
	  # predict gender
      im = caffe.io.load_image(filename, color=False)
      self.predictGender(lenet, im, self.lenetResult)
      self.predictGender(alexnet, im, self.alexnetResult)
      self.predictGender(maxout, im, self.maxoutResult)
      self.predictGender(elu, im, self.eluResult)
      self.predictGender(elu_oversampled, im, self.eluOversampledResult)
      self.displayFingerprint(filename)
      
      
  

if __name__=='__main__':
  root = Tkinter.Tk()
  GenderDeterminatorFrame(root).grid()
  root.title('Gender Determinator')
  root.geometry('375x300+0+0')
  root.mainloop()


