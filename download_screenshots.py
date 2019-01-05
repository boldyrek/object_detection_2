import numpy as np
import re
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')
# This is needed to display the images.
from utils import label_map_util

from utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'arrow_graph_100000'
MODEL_NAME = 'inference_full-5047'
MODEL_FILE = MODEL_NAME + '.tar.gz'

PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'object-detection.pbtxt')
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  print(im_width, im_height) 
  #print(list(image.getdata()))
  #print(image.getdata())
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
      #( im_width, im_height, 3)).astype(np.uint8)
# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 7) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (48, 32)
def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
    # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict
import psycopg2
conn=psycopg2.connect("host=13.59.145.243 user=postgres password=letitgo123 dbname=cars")
cur=conn.cursor()
cur.execute("select url,dir from cars,cardescription where cars.id=cardescription.carid and updated > date(now() - interval '1 day') and mileage is not null and screenshots is null" )
list_of_urls=cur.fetchall()
list_of_urls[0]
url=list_of_urls[10][0]
print(url)

def detect_object(url,directory):
    from PIL import Image
    from selenium import webdriver
    import time
    print('here is directory',directory)
    print("i'm at the begining")
    print(url)
    driver = webdriver.PhantomJS() # or add to your PATH
    #driver.set_window_size(1920, 1920) # optional
    print("i'm getting a page")
    #without this option phantomjs can never stop in almost all cases
    driver.set_page_load_timeout(40)
    #driver.maximize_window() #it is better not activate is becuase of this it cannot find elements very often
    driver.set_window_size(1500, 1500)

    print(driver.get_window_size())
    try:
        driver.get(url)
    except:
        pass
    time.sleep(5)
    print('im after sleep and driver.get')
    driver.save_screenshot(directory+'screenshot.png')
    image = Image.open(directory+'screenshot.png')
    #image = image.crop((0,0,1500,1500))
    image =image.convert('RGB')
    #print(image)
    image.save(directory+'/screenshot.jpg')
    image = Image.open(directory+'/screenshot.jpg')
#image = image.convert('RGB')
#image.getdata()
# the array based representation of the image will be used later in order to prepare the
# result image with boxes and labels on it.
#image = Image.open('Foto.jpg')

    image_np = load_image_into_numpy_array(image)
# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
# Actual detection.
    print("i'm before inference")
    output_dict = run_inference_for_single_image(image_np, detection_graph)
# Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)
#print(output_dict['detection_boxes'])
    print('score:',output_dict['detection_scores'][0])
    print('detection boxes',output_dict['detection_boxes'][0])
    #plt.figure(figsize=IMAGE_SIZE)
    #plt.imshow(image_np);
    #print(output_dict)
    return driver,image_np,output_dict,image
def calculate_x_y(driver,output_dict,image):    
    print(driver.get_window_size())
    width,length=image.size
    print(width,length)
    x=width*output_dict['detection_boxes'][0][1]
    print(output_dict['detection_boxes'][0][0],output_dict['detection_boxes'][0][1])
    y=length*output_dict['detection_boxes'][0][0]
    print('upper x and y',int(x),int(y))
    x1=width*output_dict['detection_boxes'][0][3]
    y1=length*output_dict['detection_boxes'][0][2]
    print('lower x and y',int(x1),int(y1))
    center_x=x+(x1-x)/2
    center_y=y+(y1-y)/2
    return center_x,center_y,driver
def click_display(x,y,driver,directory):
    from PIL import Image
    import time
    #as per https://stackoverflow.com/questions/5253822/why-does-document-elementfrompoint-return-null-for-elements-outside-visible-docu
    #script="document.elementFromPoint({0} - window.pageXOffset, {1} - window.pageYOffset).click();".format(str(int(x)),str(int(y)))
    script="document.elementFromPoint({0}, {1}).click();".format(str(int(x)),str(int(y)))

    print(script)
    #driver.maximize_window()
    for i in range(10):
        try:
            driver.execute_script(script)
            time.sleep(1)
        except Exception as e:
            print(e)
            pass
        i=str(i)
        driver.save_screenshot(directory+'screenshot'+i+'.png')
        image = Image.open(directory+'screenshot'+i+'.png')
        image =image.convert('RGB')
        image.save(directory+'/screenshot'+i+'.jpg')
    image = Image.open(directory+'/screenshot2.jpg')
    image_np = load_image_into_numpy_array(image)
    #plt.figure(figsize=(80, 60), dpi=80,)
    #driver.quit()
    #plt.figure(figsize=IMAGE_SIZE)
    #plt.imshow(image_np);
    #plt.show()
    return image_np
for url in list_of_urls:
    directory=url[1].strip()
    url=url[0]
#    url='http://www.highlandgm.com/VehicleDetails/used-2012-Chevrolet-Silverado_1500-LT-Aurora-ON/3297610103'
    print(directory)
    if not 'boldyrek' in directory:
        directory='/home/boldyrek/tmp_boldyrek/tmp'+directory
    else:
        directory=re.sub('tmp_boldyrek','tmp_boldyrek/tmp',directory)
    #url='http://www.allanparkmotors.com/used/vehicle/2012-smart-fortwo-pure-id8841461.htm'
    #url='https://www.lakeridgechrysler.com/used/Subaru/2017-Subaru-WRX-aea7bea00a0e0ae73b0b645fcb5ca8dc.htm'
    #url='https://www.newmarketinfiniti.com/used/INFINITI/2013-INFINITI-JX35-ef9acf830a0e0a3a2274b7ffe8daf234.htm'
    #url='https://www.boltonnissan.ca/used/Nissan/2018-Nissan-Rogue-bdffe1400a0e08ba3dfd555b53655ac4.htm'
    #url='https://www.kanatamazda.com/new/vehicle/2019-mazda-cx-3-gs-id9084650.htm'
    #url='https://www.buddsbmw.com/used/vehicle/2019-bmw-x5-xdrive50i-id9075584.htm'
    #url='https://www.allistonvw.com/en/inventory/used/2016-volkswagen-jetta-sedan-alliston-ontario/16159928/'
    #url='https://www.newmarketvolvo.com/en/used-inventory/volvo/xc90_hybrid/2017-volvo-xc90_hybrid-id5784183'
    #url='https://www.bmwgrandriver.com/used/vehicle/2018-bmw-x5-xdrive35i-id9103629.htm'
    #url='https://www.citybuick.com/used/vehicle/2017-nissan-altima-25-sv-id9090926.htm'
    driver,image_np,output_dict,image=detect_object(url,directory)
    x,y,driver=calculate_x_y(driver,output_dict,image)
    image_np=click_display(x,y,driver,directory)
    #raw_input('Continue press any key')
    cur.execute("update cars set screenshots=True where url='{0}'".format(url))  
    conn.commit()
    with open('screenshots.log','a') as f:
        f.write(url+','+directory+'\n')
f.close()
    
