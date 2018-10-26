import cv2
import matplotlib.pyplot as plt
import os

def displayImage(image):
    cv2.imshow('image' , image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def convertToCannyEdge(image):
    canny_edge_image = cv2.Canny(image , 100 , 100)
    return canny_edge_image

def createNewCannyEdgeImage(image_name , inp_path , out_path):
    image = cv2.imread(os.path.join(inp_path , image_name) , 0)   
    canny_edge_image = convertToCannyEdge(image)
    if not cv2.imwrite(os.path.join(out_path , image_name) , canny_edge_image):
        print("Error %s" %image_name)

def createNewGrayScale(image_name , inp_path , out_path):
    image = cv2.imread(os.path.join(inp_path , image_name) , 0)
    if not cv2.imwrite(os.path.join(out_path , image_name) , image):
        print("Error %s" %image_name)

# for i in range(10):
#     inp_path = '../Sign-Language-Digits-Dataset/Dataset/' + str(i)
#     files = list(os.walk(inp_path))[0][2]
#     outgray = '../Sign-Language-Digits-Dataset/GrayScaleDataset/' + str(i) 
#     outcanny = '../Sign-Language-Digits-Dataset/CannyEdgeDataset/' + str(i)
#     for img_name in files:
#         createNewCannyEdgeImage(img_name , inp_path , outcanny)
#         createNewGrayScale(img_name , inp_path , outgray)



