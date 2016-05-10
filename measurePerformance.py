#
# Measures performance of some set given the model classifier
#
import caffe
import numpy as np
from os.path import basename

#load the model
#net = caffe.Net('model/elu_deploy.prototxt',
#                'model/SAVE/elu_iter_10000_augmented.caffemodel',
#                caffe.TEST)
#net2 = caffe.Classifier('model/elu_deploy.prototxt','model/SAVE/elu_iter_10000_1.caffemodel', image_dims=(256,256))
net2 = caffe.Classifier('model/elu_oversampled_deploy.prototxt','model/SAVE/elu_iter_10000_augmented.caffemodel', image_dims=(256,256))

# load input and configure preprocessing
# for models not 
#transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
#transformer = caffe.io.Transformer({'data': net2.blobs['data'].data.shape})
#transformer = caffe.io.Transformer({'data': net3.blobs['data'].data.shape})

#transformer.set_transpose('data', (2,0,1))

#transformer.set_channel_swap('data', (2,1,0))
#transformer.set_raw_scale('data', 255.0)

#note we can change the batch size on-the-fly
#since we classify only one image, we change batch size from 10 to 1
#net.blobs['data'].reshape(1,1,227,227)
#net2.blobs['data'].reshape(1,1,227,227)
#net3.blobs['data'].reshape(1,1,256,256)

infoFile = open('model/test_list.txt', 'r')
#infoFile = open('model/test2.txt', 'r')
i=0
FingerTypes = ['R_Thumb', 'R_Index', 'R_Middle', 'R_Ring','R_Little','L_Thumb','L_Index', 'L_Middle', 'L_Ring', 'L_Little']
FingerTypeCounts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

FingerTypeMale = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
FingerTypeFemale = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
FingerTypeMaleCorrect = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
FingerTypeFemaleCorrect = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for line in infoFile:
    i=i+1
	
    imageFile = line.split('\t')[0]
    tag = int(line.split('\t')[1].rstrip())
    fingertype = basename(imageFile).split('.')[0].split('_')[1]
	
	
    if tag == 1 :
        FingerTypeMale[int(fingertype)-1] = FingerTypeMale[int(fingertype)-1] + 1
    else :
        FingerTypeFemale[int(fingertype)-1] = FingerTypeFemale[int(fingertype)-1] + 1
		
    print '==========' + str(i) + '===' + imageFile
	#load the image in the data layer
    im = caffe.io.load_image(imageFile, color=False)
    #im = caffe.io.resize( im, (256,256))
    #net.blobs['data'].data[...] = transformer.preprocess('data', im)
	#net2.blobs['data'].data[...] = transformer.preprocess('data', im)
	#net3.blobs['data'].data[...] = transformer.preprocess('data', im)
    out = net2.predict(np.array([im]))
    print str(out) + ' ' + str(tag)
    
	#compute
    #out = net.forward()
	#out2 = net2.forward()
	#out3 = net3.forward()
	# other possibility : out = net.forward_all(data=np.asarray([transformer.preprocess('data', im)]))
	
	# compute probability
    femaleProb = out[0][0] #(out['prob'][0][0] + out2['prob'][0][0])/2
    maleProb = out[0][1] #(out['prob'][0][1] + out2['prob'][0][1])/2
    predictedClass = 1
    if femaleProb >= maleProb :
        predictedClass = 0
	

    # predict class
    if predictedClass == tag :
        if tag==1 :
            FingerTypeMaleCorrect[int(fingertype)-1] = FingerTypeMaleCorrect[int(fingertype)-1] + 1
        else :
            FingerTypeFemaleCorrect[int(fingertype)-1] = FingerTypeFemaleCorrect[int(fingertype)-1] + 1


infoFile.close()

# Print results
print 'Results'
print FingerTypeFemale
print FingerTypeMale

Male=0
Female=0
MaleCorrect=0
FemaleCorrect=0
i=0
while i<len(FingerTypeMale) : 
    print FingerTypes[i] + " = " + str(FingerTypeMaleCorrect[i] + FingerTypeFemaleCorrect[i]) + '/' + str(FingerTypeMale[i] + FingerTypeFemale[i]) + ' (' + str((FingerTypeMaleCorrect[i]+FingerTypeFemaleCorrect[i] ) * 100.0 / (FingerTypeMale[i]+FingerTypeFemale[i])) + ')'
    print 'Males = ' + str(FingerTypeMaleCorrect[i]) + '/' + str(FingerTypeMale[i]) +  '(' + str(FingerTypeMaleCorrect[i] * 100.0 / FingerTypeMale[i]) + ')'
    print 'Females = ' + str(FingerTypeFemaleCorrect[i]) + '/' + str(FingerTypeFemale[i]) +  '(' + str(FingerTypeFemaleCorrect[i] * 100.0 / FingerTypeFemale[i]) + ')'
    Male = Male + FingerTypeMale[i]
    print Male
    Female = Female + FingerTypeFemale[i]
    MaleCorrect = MaleCorrect + FingerTypeMaleCorrect[i]
    FemaleCorrect = FemaleCorrect + FingerTypeFemaleCorrect[i]
    print '-------------------------------------'
    i = i+1
    
print 'Fingerprint Gender Stats'
print '    Male = ' + str(MaleCorrect) + '/' + str(Male) + '(' + str(MaleCorrect*100.0/Male) + ')'
print '  Female = ' + str(FemaleCorrect) + '/' + str(Female) + '(' + str(FemaleCorrect*100.0/Female) + ')'

print '    Overall = ' + str(FemaleCorrect + MaleCorrect) + '/' + str(Female + Male) + '(' + str((FemaleCorrect+MaleCorrect)*100.0/(Female+Male)) + ')'
print '======================='
