import cv2

# camera settings and params
cap = cv2.VideoCapture(0)

# cap.set(width, height)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)

# run with image instead:
# img = cv.imread('filename')

# import all names from coco
c_names= []
c_file = 'coco.names'
with open(c_file,'rt') as f:
    c_names = f.read().rstrip('\n').split('\n')

# write paths
p_config = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
p_weights = 'frozen_inference_graph.pb'

# creating model with opencv
model = cv2.dnn_DetectionModel(p_weights,p_config)

# configs from documentation
model.setInputSize(320,320)
model.setInputScale(1.0/ 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# object detection threshold
cutoff = 0.70

# let model process img and make predictions
while True:
    success,img = cap.read() # define image
    c_ids, confid, bound = model.detect(img,confThreshold=cutoff)
    print(c_ids,bound) #[[index]] [[param1 param2 param3 param4]]

    if (len(c_ids) != 0):
        for class_id, conf, box in zip(c_ids.flatten(),confid.flatten(),bound):
            cv2.rectangle(img,box,color=(245, 255, 54),thickness=3)

            cv2.putText(img,c_names[class_id-1].upper(),(box[0]+10,box[1]+30), # text positioning
                        cv2.FONT_HERSHEY_PLAIN,1,(245, 255, 54),2) # font
            
            cv2.putText(img,str(round(conf*100,2)),(box[0]+200,box[1]+30), # gives percentage
                        cv2.FONT_HERSHEY_PLAIN,1,(245, 255, 54),2)

    cv2.imshow("Output",img) #visualisation
    cv2.waitKey(1)
