import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
import zipfile
from PIL import Image
from object_detection.utils import ops as utils_ops
from utils import label_map_util
import psycopg2
import time

class CarImageDownloader():


    def __init__(self):

        MODEL_NAME='inference_full-5047'
        PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
        PATH_TO_LABELS=os.path.join('data','object-detection.pbtxt')
        category_label = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def= tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph=fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def,name='')


    def load_image_into_numpy_array(self,image):

         (im_width,im_height)=image.size
         return np.array(image.getdata()).reshape(
              (im_height,im_width,3)).astype(np.uint8)


    def run_inference_for_single_image(self,image,graph):

         with graph.as_default():
              with tf.Session() as sess:
                   ops = tf.get_default_graph().get_operations()
                   all_tensor_names= {output.name for op in ops for output in op.outputs}
                   tensor_dict= {}
                   for key in [
                   'num_detections','detection_boxes', 'detection_scores',
                   'detection_classes','detection_masks'
                     ]:
                       tensor_name= key + ':0'
                       if tensor_name in all_tensor_names:
                            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
                   if 'detections_masks' in tensor_dict:
                         detection_boxes=tf.squeeze(tensor_dict['detection_boxes'],[0])
                         detection_masks=tf.squeeze(tensor_dict['detection_masks'],[0])
                         real_num_detection= tf.cast(tensor_dict['num_detections'][0],tf.int32)
                         detection_boxes = tf.slice(detection_boxes, [0,0], [real_num_detection, -1])
                         detection_masks = tf.slice(detection_masks, [0,0,0], [real_num_detection,-1,-1])
                         detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed,0.5),tf.uint8)
                         tensor_dict['detection_masks']=tf.expand_dims(detection_masks_reframed,0)
                   image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
                   output_dict=sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image,0)})
                   output_dict['num_detections']= int(output_dict['num_detections'][0])
                   output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                   output_dict['detection_scores'] = output_dict['detection_scores'][0]
                   if 'detection_masks' in output_dict:
                       output_dict['detection_masks']=output_dict['detection_masks'][0]
                   return output_dict


    def get_urls(self):

        conn=psycopg2.connect('user=postgres host=13.59.145.243 password=letitgo123 dbname=cars')
        cur=conn.cursor()
        cur.execute("select url,dir from cars,cardescription where cars.id=cardescription.carid and updated > date(now() - interval '1 day') and mileage is not null")
        list_of_urls=cur.fetchall()
        print('returning list of urls')
        return list_of_urls


    def detect_object(self,url):

        from PIL import Image
        from selenium import webdriver
        self.driver=webdriver.PhantomJS()
        self.driver.set_page_load_timeout(40)
        try:
          self.driver.get(url)
        except : 
          pass
        time.sleep(5)
        self.driver.save_screenshot('screenshot.png')
        image=Image.open('screenshot.png')
        image=image.convert('RGB')
        image.save('screenshot.jpg')
        image_np=self.load_image_into_numpy_array(image)
        image_np_expanded=np.expand_dims(image_np,axis=0)
        self.output_dict=self.run_inference_for_single_image(image_np,self.detection_graph)

        return image


    def calculate_x_y(self,image):
    
        width,length = image.size
        x=width*self.output_dict['detection_boxes'][0][1]
        y=length*self.output_dict['detection_boxes'][0][0]
        x1=width*self.output_dict['detection_boxes'][0][3]
        y1=length*self.output_dict['detection_boxes'][0][2]
        center_x=(x1-x)/2
        center_y=(y1-y)/2
        return center_x,center_y



    def click_display(self,x,y):
        
        script='document.ElementFromPoint.click({0},{1})'.format(str(int(x)),str(int(y)))    
        for i in range(5):
            try:
                driver.execute_script(script)
                time.sleep(1)
            except Exception as e:
                print(e)
                pass
        driver.save_screenshot('screenshot'+i+'.png')
        Image.open('screenshot'+i+'.png')
        image=image.convert('RGB')
        image.save('screenshot'+i+'.jpg')


def main():

    downloader=CarImageDownloader()
    urls=downloader.get_urls()
    for url in urls:
         image=downloader.detect_object(url)     
         x,y=downloader.calculate_x_y(image)
         downloader.click_display(x,y)
main()             
