# Recognises faces using classification algorithm like KNN 

# 1. Read video stream using opencv
# 2. Extract faces out of it 
# 3. load the training data (numpy arrays of all the person
      # x-values stored in numpy arrays
      # y- values we need to assign to each person
# 4. use knn to find prediction of face 
# 5. map the predicted id to name of user
# 6. Display the predictions on the screen - bounding box and name 

import numpy as np 
import cv2
import os
## KNN CODE

def distance(v1,v2):
	#Euclidean
	return np.sqrt(((v1-v2)**2).sum())
def knn(train,test,k=5):
	dist=[]
	for i in range(train.shape[0]):
		# Get the vector and label
		ix = train[i, :-1]
		iy = train[i, -1]
		# Compute the distance from test point
		d = distance(test, ix)
		dist.append([d, iy])
	# Sort based on distance and get top k
	dk = sorted(dist,key=lambda x:x[0])[:k]
	# Retrieve only the labels
	labels = np.array(dk)[:, -1]
	# Get frequencies of each label
	output = np.unique(labels, return_counts=True)
	# Find max frequency and corresponding label
	index = np.argmax(output[1])
	return output[0][index]

	
	
	    
        
		
       
    
       

       
       
  
 ## END OF KNN

 # Face detection
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
skip=0
face_data=[]
dataset_path='./data/'
labels=[]


class_id=0  # labels for the given file
names ={} #dictionary for creating mapping between  names and id
for fx in os.listdir(dataset_path):
	if fx.endswith('.npy'):
		data_item = np.load(dataset_path+fx)
		names[class_id]=fx[:-4] # -4 to avoid .npy and map only name of person with class_id
		print("Loaded "+fx)
		face_data.append(data_item)
		# create labels for the class 
		target = class_id*np.ones((data_item.shape[0]),)
		class_id+=1
		labels.append(target)

face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis=0).reshape((-1,1))

trainset = np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape) #(64,30001)- 30000 fore features 1-labels

## testing 
while (1):
	ret,frame = cap.read()

	if(ret==0):
		continue
	faces = face_cascade.detectMultiScale(frame,1.3,5)
	
	for face in faces:
		x,y,w,h = face
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))
		# predicted label(output)
		out = knn(trainset,face_section.flatten())
		pred_name = names[int(out)]

		cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
	cv2.imshow("Faces",frame)
	key_pressed = cv2.waitKey(1)&0xFF
	
	if(key_pressed)==ord('q'):
	   break
cap.release()
cv2.destroyAllWindows()	
    
   	
        

