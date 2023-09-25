# @title Final Project
 # -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 19:06:22 2023

@author: omara
"""
import keras.backend as K
from contextlib import redirect_stdout
import io
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
from scipy.io import loadmat
import os
import random
import numpy as np
from tensorflow.keras import layers,models,optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from keras.utils import to_categorical
from numpy import asarray, savez_compressed, load
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from skimage.morphology import medial_axis
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_dilation
from skimage.morphology import binary_erosion
from skimage.morphology import skeletonize
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout,Concatenate
from keras.initializers import glorot_uniform
from tensorflow.keras import initializers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from keras.initializers import RandomNormal
from keras.callbacks import ReduceLROnPlateau,Callback
import math
import re
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'




#%%is image uni color
def is_image_useless(image):


    image_array = np.array(image)
    white_percentage = 0.96
    if len(image_array.shape) > 2:

      image_array = np.mean(image_array, axis=-1)




   # if not contains_black_pixels:# or not square_is_not_empty:
     # print(f"no black pixels")
     # return True
    tolerance = 255

    max_white_pixels = image_array.shape[0] * image_array.shape[1]
    max_white_pixels *=  white_percentage

    image_array[image_array < tolerance] = 0


    num_white_pixels = np.count_nonzero(image_array)


    if num_white_pixels == 0:

        return True



    if num_white_pixels < max_white_pixels:
            return False  # The image is not unicolor

    return True  # The image is unicolor

#%%line density
def get_line_density(line):

  line_array = np.array(line)
  tolerance = 255

  line_array[line_array < tolerance] = 0

  num_white_pixels = np.count_nonzero(line_array)
  pixels_colors = line_array.flatten()
  black_pixels_density = 1 - (num_white_pixels/len(pixels_colors))
  return black_pixels_density
#%%shuffle
def shuffle_samples(squares,labels,image_directory,num_classes):

    num_samples = len(squares)
    squares = np.array(squares)
    label_map = retrieve_image_labels(image_directory,num_classes)

    # Use list comprehension to convert the string labels to numeric labels
    numeric_labels = np.array([label_map[label] for label in labels])
    # Generate a random permutation of indices
    shuffled_indices = np.random.permutation(num_samples)

    # Use the shuffled indices to shuffle both squares and labels
    shuffled_squares = np.array(squares[:len(squares)])
    shuffled_labels = np.array(numeric_labels[:len(squares)])
    # shuffled_labels = np.array(numeric_labels[:len(squares)])#shuffled_indices])
    return shuffled_squares,shuffled_labels

#%%model architecture
def AlexNet(square_size,num_classes):

    input_shape = (square_size,square_size,1)
    inp = layers.Input((square_size, square_size, 1))
    x = Conv2D(48, kernel_size=(11,11), strides= 4,
                        padding= 'valid', activation= 'relu',
                        input_shape= input_shape,
                        kernel_initializer= 'he_normal')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2), strides= (2,2),
                          padding= 'valid', data_format= None)(x)

    x = Conv2D(256, kernel_size=(5,5), strides= 1,
                   padding= 'same', activation= 'relu',
                   kernel_initializer= 'he_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2), strides= (2,2),
                          padding= 'valid', data_format= None)(x)

    x = Conv2D(512, kernel_size=(3,3), strides= 1,
                    padding= 'same', activation= 'relu',
                    kernel_initializer= 'he_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2), strides= (2,2),
                          padding= 'valid', data_format= None)(x)

    x = Flatten()(x)

    x = Dense(1024, activation= 'relu')(x)
    x = Dense(1024, activation= 'relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(num_classes, activation= 'softmax')(x)

   
    model = Model(inputs=inp, outputs=x)
    return model




#%%plotting

def plot_graph(train_param, val_param, title, xlabel, ylabel):
    plt.plot(train_param)
    plt.plot(val_param)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.show()

def plot_loss(history):
    loss_params = {
        'train_param' : history.history['loss'],
        'val_param' : history.history['val_loss'],
        'title' : 'Loss - Epochs',
        'xlabel' : 'Epochs',
        'ylabel' : 'Loss'
        }
    plot_graph(**loss_params)

def plot_accuracy(history):
    accuracy_params = {
        'train_param' : history.history['accuracy'],
        'val_param' : history.history['val_accuracy'],
        'title' : 'Accuracy - Epochs',
        'xlabel' : 'Epochs',
        'ylabel' : 'Accuracy'
        }
    plot_graph(**accuracy_params)

#%%classify patches
def classify_patches(model, squares):
    class_counts = [0] * model.output_shape[-1]
    #display_images_slides(squares)
    #display_images_slides(squares,len(squares),False)
    dummy_stream = io.StringIO()
    #print(f"len line: {len(squares)}")
    # Use the context manager to suppress stdout
    with redirect_stdout(dummy_stream):
      for square in squares:
          square = np.expand_dims(square, axis=-1)
          square = np.expand_dims(square, axis=0)
          predictions = model.predict(square)
          
          predicted_class = np.argmax(predictions)
          predicted_class = int(predicted_class)

          class_counts[predicted_class] += 1

    test_line_class = np.argmax(class_counts)

    return test_line_class

#%%thick lines
def thicken_lines(binary_image, iterations=1):
    # Perform dilation on the binary image
    #dilated_image = binary_dilation(binary_image, iterations=iterations)
   eroded_image = binary_image.copy()
   for _ in range(iterations):
       eroded_image = binary_erosion(eroded_image)

   return eroded_image
#%%thin lines
def thinner_lines(binary_image, iterations = 1):
    dilated_image = binary_dilation(binary_image, iterations=iterations)
    return dilated_image
#%% apply filter
def apply_filter(original_squares, filter_function, labels = [], *args, **kwargs):
    filtered_squares = [filter_function(patch, *args, **kwargs) for patch in original_squares]

    filtered_squares = [binary_image.astype(np.uint8) * 255 for binary_image in filtered_squares]
    if len(labels) > 0 :
        filtered_labels = labels[:len(filtered_squares)]
        return filtered_squares,filtered_labels
    else:
        return filtered_squares

#%%filtered squares
def get_filtered_squares(squares,labels = [],thicken=7,thinner=3):
    filtered_squares = apply_filter(squares,thicken_lines, iterations = thicken)
    new_squares = apply_filter(filtered_squares,thinner_lines, iterations = thinner)
    print("things did get filtered")
    if len(labels) > 0:
      squares = np.concatenate((squares,new_squares),axis = 0)
      filtered_labels = labels[:len(filtered_squares)]
      labels = np.concatenate((labels,filtered_labels),axis = 0)
      return squares,labels
    return new_squares

#%%data gen
def custom_data_generator(images, labels, batch_size, datagen):
    num_samples = len(images)
    indices = np.arange(num_samples)
    while True:
        # Shuffle the data indices at the beginning of each epoch
        np.random.shuffle(indices)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            batch_images = images[batch_indices]
            batch_labels = labels[batch_indices]

            # Generate FFT features for the batch of images
            #batch_fft_features = fft_images(batch_images)  # Modify this according to your function

            # Apply data augmentation to the original images
            augmented_images = datagen.flow(batch_images, batch_size=batch_size, shuffle=False)
            augmented_images = next(augmented_images)

            yield augmented_images, batch_labels
#%%display_image_slides
def display_images_slides(image_array,num_images_to_display = 10,shuffle = True):


    # Randomly select 'num_images_to_display' indices from the image array
    if shuffle:
      random_indices = random.sample(range(len(image_array)), num_images_to_display)
    else:
      random_indices = list(range(len(image_array)))
    i  = 0
    for idx in random_indices:
        if i == num_images_to_display:
          break
        i+=1
        img = image_array[idx]
        # Calculate the color difference between all pixels and the first pixel
        color_difference = np.abs(img - img[0, 0])
        # Check if any pixel exceeds the tolerance
        non_zero_count = np.count_nonzero(color_difference)

        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Image {idx + 1}, Non-Zero Pixels: {non_zero_count}")
        plt.show()
#%%class samples counter
def count_strings(array):
    counts = {}
    for item in array:
        if item in counts:
            counts[item] += 1
        else:
            counts[item] = 1
    return counts
#%% smaller squares
def get_smaller_squares(squares,smaller_size,max_width=0,max_height=0):
    #padded_squares = pad_and_center_images(max_width,max_height,squares)
    smaller_squares = []
    new_size = (smaller_size,smaller_size)
    for square in squares:
        pil_image = Image.fromarray(square.astype('uint8'))  # Convert NumPy array to PIL Image
        resized_image = pil_image.resize(new_size, Image.LANCZOS)  # Resize
        smaller_squares.append(np.array(resized_image))  # Convert PIL Image back to NumPy array

    return smaller_squares


#%%data prep

def thresholding(image):

    ret,thresh = cv2.threshold(image,80,255,cv2.THRESH_BINARY_INV)

    return thresh
#%%get useful lines
def get_useful_lines(image_path,line_pos_path):
    stam = loadmat(line_pos_path)
    lines_y = stam['peaks_indices'].flatten()

    filename = os.path.splitext(os.path.basename(image_path))[0]

    img = Image.open(image_path)
    width,height = img.size


    indices = []
    for i in range(len(lines_y) - 1):
        line_start = lines_y[i] * 5
        line_end = lines_y[i + 1]*5

        line = img.crop((0, line_start, width, line_end))
        if not is_image_useless(line) and not is_test_line(line_start,line_end,line_pos_path):
            indices.append(i)

    return indices
#%%
def is_test_line(line_start,line_end,line_pos_path):
  stam = loadmat(line_pos_path)
  top_test_y = stam['top_test_area'].flatten()[0]
  bot_test_y = stam['bottom_test_area'].flatten()[0]
  if (abs(top_test_y - line_start) > 60) and (abs(bot_test_y - line_end) > 60):
        return False
  else:

    return True
#%%

def get_words_from_pages(num_classes,image_directory,line_positions_directory,test_line_directory,val_line_directory):
    all_tsquares = []
    all_tlabels = []

    for i,filename in enumerate(os.listdir(image_directory)[:num_classes]):
        filename = os.path.splitext(filename)[0]
        tsquares,tlabels = cut_lines_into_words(image_directory+'/'+filename+'.jpg',
                        test_line_directory+'/'+filename+'.jpg',
                        line_positions_directory+'/'+filename+'.mat',
                        val_line_directory+'/'+filename+'.jpg'
                        )
        all_tsquares.extend(tsquares)
        all_tlabels.extend(tlabels)



    return  all_tsquares,all_tlabels
#%%
def get_most_dense_lines(image_path,line_pos_path,useful_indices):

    filename = os.path.splitext(os.path.basename(image_path))[0]
    img = cv2.imread(image_path)
    img_height, img_width, img_color = img.shape
    stam = loadmat(line_pos_path)
    lines_y = stam['peaks_indices'].flatten()
    useful_lines = [img[lines_y[i]*5:lines_y[i+1]*5,0:img_width] for i in useful_indices if not is_test_line(lines_y[i]*5, lines_y[i+1]*5,line_pos_path)]
  
    num_needed_lines =1
    index = 0
    max_density = 0
    for i,line in enumerate(useful_lines):
      if get_line_density(line) > max_density:
        index = useful_indices[i]
        max_density = get_line_density(line)
    dense_indices = index
 

    return dense_indices

#%%
def cut_lines_into_words(image_path,test_line_path,line_pos_path,val_line_path):

    img = cv2.imread(image_path)
    img_height, img_width, img_color = img.shape
    difference_ratio = 1
    filename = os.path.splitext(os.path.basename(image_path))[0]
    if img_width > 1000:

        new_width = 1000
        aspect_ratio = img_width/img_height
        new_height = int(new_width/aspect_ratio)
        height_difference_ratio = new_height/img_height
        width_difference_ratio = new_width/img_width
        img_width = new_width
        img = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_AREA)


    thresh_img = thresholding(img)

    kernel = np.ones((3,10), np.uint8)
    dilated = cv2.dilate(thresh_img, kernel, iterations = 1)


    img3 = img.copy()
    train_words_list = []

    stam = loadmat(line_pos_path)


    org_img = cv2.imread(image_path)
    lines_y = stam['peaks_indices'].flatten()
    top_test_y = stam['top_test_area'].flatten()[0]
    bot_test_y = stam['bottom_test_area'].flatten()[0]
    
    useful_indices = get_useful_lines(image_path, line_pos_path)

    dense_lines_indices = get_most_dense_lines(image_path,line_pos_path,useful_indices)

  
    total_width = 0
    for i in range(len(lines_y) - 1):


        line_start = lines_y[i] * 5
        line_end = lines_y[i + 1]*5
        if i in useful_indices and i != dense_lines_indices:

            line_start *=height_difference_ratio
            line_end *=height_difference_ratio

            x = 0
            y = int(line_start)
            w = int(img_width)
            h = int(line_end-line_start)

            roi_line = dilated[y:y+h, x:x+w]
            roi_line = cv2.cvtColor(roi_line, cv2.COLOR_BGR2GRAY)


            # draw contours on each word
            (cnt, heirarchy) = cv2.findContours(roi_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            sorted_contour_words = sorted(cnt, key=lambda cntr : cv2.boundingRect(cntr)[0])

            for word in sorted_contour_words:
                x2, y2, w2, h2 = cv2.boundingRect(word)
                area = w2*h2

                if area < 400 or is_image_useless(img3[y+y2:y+y2+h2,x+x2:x+x2+w2]):

                  continue
                total_width += w2

                train_words_list.append([x+x2, y+y2, x+x2+w2, y+y2+h2])
          
        elif i == dense_lines_indices:
            val_line = org_img[line_start:line_end,0:int(img_width/width_difference_ratio)]

        elif is_test_line(line_start,line_end,line_pos_path):
            test_line = org_img[line_start:line_end,0:int(img_width/width_difference_ratio)]



    num_samples = len(train_words_list)
    avg_width = total_width/num_samples
    cv2.imwrite(test_line_path, test_line)
    cv2.imwrite(val_line_path, val_line)

    train_squares = crop_squares(train_words_list, org_img, width_difference_ratio, height_difference_ratio, avg_width)
    train_labels = [filename] * len(train_squares)

    
    return train_squares,train_labels#,val_squares,val_labels
################################################################################
def crop_squares(words_list, org_img, width_difference_ratio, height_difference_ratio, avg_width):
    squares = []
    print(f"original len in crop: {len(words_list)}")
    for word in words_list:
        x = int(word[0] / width_difference_ratio)
        y = int(word[1] / height_difference_ratio)
        x2 = int(word[2] / width_difference_ratio)
        y2 = int(word[3] / height_difference_ratio)
        width = x2 - x
        min_cut = avg_width*4000
    
        if  width > min_cut :
            num_crops = 4

            step_size = (x2 - x) // num_crops


            for i in range(num_crops):
                new_x = x + i * step_size
                new_x2 = x + (i + 1) * step_size
                word_square = org_img[y:y2, new_x:new_x2]


                squares.append(word_square)
        else:
            word_square = org_img[y:y2, x:x2]
            squares.append(word_square)
    print(f" len in end of crop: {len(squares)}")
    return squares
#%%
def cut_test_line_into_words(test_line_path,line_pos_path,get_val = False):

    img = cv2.imread(test_line_path)
    img_height, img_width, img_color = img.shape
    filename = os.path.splitext(os.path.basename(test_line_path))[0]


    if img_width > 1000:

         new_width = 1000
         aspect_ratio = img_width/img_height
         new_height = int(new_width/aspect_ratio)
         height_difference_ratio = new_height/img_height
         width_difference_ratio = new_width/img_width
         img_width = new_width
         img = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_AREA)


    thresh_img = thresholding(img)

    kernel = np.ones((3,10), np.uint8)
    dilated = cv2.dilate(thresh_img, kernel, iterations = 1)


    img3 = img.copy()
    words_list = []
    line_start = 0
    line_end = 0
    stam = loadmat(line_pos_path)
    org_img = cv2.imread(test_line_path)
   # if not get_val:
    top_test_y = stam['top_test_area'].flatten()[0]
    bot_test_y = stam['bottom_test_area'].flatten()[0]
    line_end = abs(bot_test_y-top_test_y)

    roi_line = dilated

    x = 0
    y = int(line_start)
    w = int(img_width)
    h = int(line_end-line_start)
    roi_line = cv2.cvtColor(roi_line, cv2.COLOR_BGR2GRAY)

    # draw contours on each word
    (cnt, heirarchy) = cv2.findContours(roi_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contour_words = sorted(cnt, key=lambda cntr : cv2.boundingRect(cntr)[0])
    total_width = 0
    for word in sorted_contour_words:
        x2, y2, w2, h2 = cv2.boundingRect(word)
        area = w2 * h2

        if area < 400 or is_image_useless(img3[y+y2:y+y2+h2,x+x2:x+x2+w2]):
            continue

        total_width += w2
        words_list.append([x+x2, y+y2, x+x2+w2, y+y2+h2])


    avg_width = total_width/(len(words_list))
    word_squares = crop_squares(words_list, org_img, width_difference_ratio, height_difference_ratio, avg_width)

    return word_squares

#%%
def get_val_lines(val_directory,line_positions_directory,smaller_square):
    val_lines = []
    label_dict = retrieve_image_labels(val_directory,num_classes)
    i = 0
    for filename in label_dict:

        print(f"filename in get val lines: {filename}")
        line_pos = line_positions_directory + '/' + filename +'.mat'
        img_path = val_directory + '/' + filename +'.jpg'
        val_squares = cut_test_line_into_words(img_path,line_pos,get_val = True)

        #val_squares = get_filtered_squares(val_squares,0,0)
        #display_images_slides(val_squares,1)
        print(i)
        val_squares = get_smaller_squares(val_squares, smaller_square)
        val_squares = np.array(val_squares)
        val_squares = np.mean(val_squares, axis=-1)
        val_squares = val_squares.reshape(-1, smaller_square, smaller_square, 1)

        val_lines.append(val_squares)
        i+=1
    return val_lines
#%%
class WriterPredictionCallback(Callback):
    def __init__(self,val_lines,frequency):
        super(WriterPredictionCallback, self).__init__()
        self.val_lines = val_lines
        self.frequency = frequency
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.frequency != 0:
          return
        true_predictions = []
        model_predictions = []
        for i,line in enumerate(self.val_lines):

            class_probability = classify_patches(self.model, line)
            true_predictions.append(i)
            model_predictions.append(class_probability)

        correct_sum = 0

        for true,guess in zip(true_predictions,model_predictions):

            if(guess == true):
                correct_sum+=1


        success_rate = correct_sum/len(true_predictions)
        print(f"line validation success rate = {success_rate*100}%")

#%%create label map
def retrieve_image_labels(image_directory,num_classes):

    filenames = os.listdir(image_directory)

    image_filenames = [os.path.splitext(filename)[0] for filename in filenames if filename.endswith(('.jpg', '.png'))]
    image_filenames = image_filenames[:num_classes]
    pattern = r'lines(\d+)_Page_(\d+)'

    sorted_filenames = []
    labels = []

    image_filenames.sort()

    for filename in image_filenames:
        match = re.match(pattern, filename)
        if match:
            i_value = int(match.group(1))
            j_value = int(match.group(2))
            sorted_filenames.append(filename)
            labels.append((i_value, j_value))

    unique_labels = list(set(image_filenames))
    unique_labels.sort()
    label_to_numeric = {label: index for index, label in enumerate(unique_labels)}



    return label_to_numeric

#%%
num_classes = 50
smaller_square_size = 180
# # ##################################################
image_directory = "/content/3_ImagesLinesRemovedBW"
test_line_directory = '/content/TestLines'
line_positions_directory = "/content/5_DataDarkLines"
val_line_directory = "/content/ValidationLines"
#print("alive1")
max_height = 0
max_width = 0
#del tsquares

squares,labels = get_words_from_pages(num_classes,image_directory,line_positions_directory,test_line_directory,val_line_directory)
tsquares,vsquares,tlabels,vlabels = train_test_split(
    squares, labels, test_size=0.2,stratify=labels, random_state=42
)





tsquares = get_smaller_squares(tsquares, smaller_square_size,max_width,max_height)
vsquares = get_smaller_squares(vsquares, smaller_square_size,max_width,max_height)





tsquares,tlabels = shuffle_samples(tsquares, tlabels,image_directory,num_classes)
vsquares,vlabels = shuffle_samples(vsquares, vlabels,image_directory,num_classes)

savez_compressed('data_project.npz',
                    tsquares, tlabels,
                    vsquares, vlabels)
val_lines = get_val_lines(val_line_directory,line_positions_directory,smaller_square_size)
#%% data load

dict_data = np.load('data_project.npz', allow_pickle=True)
tsquares = dict_data['arr_0'].astype(np.uint8)
tlabels = dict_data['arr_1'].astype(np.uint8)
vsquares = dict_data['arr_2'].astype(np.uint8)
vlabels = dict_data['arr_3'].astype(np.uint8)

#%%

model = AlexNet(smaller_square_size,num_classes)
lr = 0.001
lr_factor =0.8
batch_size = 128
epochs = 200
lr_patience = 7
early_stopping_patience = 15


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=lr_factor,min_delta=0.001, patience=lr_patience, min_lr=1e-6)

opt = optimizers.Adam (learning_rate = lr)

early_stopping = EarlyStopping(monitor='val_accuracy',mode='max', patience=early_stopping_patience, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(
    'best_model_weights.h5',
    save_best_only=True,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

datagen = ImageDataGenerator(
    #preprocessing_function=skeletonyze,
    rotation_range = 2,
    width_shift_range=0.4,  
    height_shift_range=0.4,  
    shear_range=0.3,  
    zoom_range=0.4,  
    fill_mode='nearest' 
)
print(f"2 squares: {len(tsquares)} , labels: {len(tlabels)}")
tsquares = np.mean(tsquares, axis=-1)
vsquares = np.mean(vsquares, axis=-1)
tsquares = tsquares.reshape(-1, smaller_square_size, smaller_square_size, 1)
vsquares = vsquares.reshape(-1, smaller_square_size, smaller_square_size, 1)



datagen.fit(tsquares)
print(f"4 squares: {len(tsquares)} , labels: {len(tlabels)}")
print(f"lr = {lr} lr_factor = {lr_factor} batch_size = {batch_size} epochs = {epochs}")
print(f"reduce_lr patience = {lr_patience} early_Stopping_patience = {early_stopping_patience}")
model.summary()
#model.load_weights('best_model_weights.h5')

model.compile(optimizer=opt,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
y_train = tlabels.flatten()
class_weights = compute_class_weight('balanced',classes = np.unique(y_train),y = y_train)
class_weights_dict = dict(zip(np.unique(tlabels), class_weights))
tlabels = to_categorical(tlabels, num_classes=num_classes)
vlabels = to_categorical(vlabels, num_classes=num_classes)
class_counts_train = np.sum(tlabels, axis=0)

class_counts_val = np.sum(vlabels, axis=0)

print("Training Class Counts:")
print(class_counts_train)

print("Validation Class Counts:")#
print(class_counts_val)

train_generator = custom_data_generator(tsquares, tlabels, batch_size, datagen)
history = model.fit(train_generator, class_weight=class_weights_dict,
                    steps_per_epoch=len(tsquares) // batch_size,  # Adjust this value based on your dataset
                    epochs=epochs,
                    validation_data=(vsquares, vlabels),
                    callbacks=[early_stopping, model_checkpoint,
                              WriterPredictionCallback(val_lines,5),reduce_lr])

plot_loss(history)
plot_accuracy(history)
model_json = model.to_json()
with open("model_project.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_project.h5")
#print("Saved model to disk")
del tsquares,vsquares,tlabels,vlabels
#%%model load
# later...

# load json and create model
#json_file = open('model_project.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#model = model_from_json(loaded_model_json)
# # del train_squares,val_squares,train_labels,val_labels
#model.load_weights("model_project.h5")


###############################################################################
#%% testing

test_lines_directory = '/content/TestLines'
true_predictions = []
model_predictions = []

for image_file in os.listdir(test_lines_directory):
    if image_file.endswith('.jpg') or image_file.endswith('.png'):
        image_path = os.path.join(test_lines_directory, image_file)
        filename = os.path.splitext(os.path.basename(image_path))[0]
        line_pos_path = line_positions_directory+'/'+filename+'.mat'
        test_squares = cut_test_line_into_words(image_path, line_pos_path)
        test_squares = get_smaller_squares(test_squares, smaller_square_size,max_width,max_height)
        test_squares = np.array(test_squares)
        test_squares = np.mean(test_squares, axis=-1)
        test_squares = test_squares.reshape(-1, smaller_square_size, smaller_square_size, 1)
        class_probability = classify_patches(model, test_squares)
        
        true_predictions.append(filename)
        model_predictions.append(class_probability)

correct_sum = 0
wrong_predictions = []
label_map = retrieve_image_labels(image_directory,num_classes)
for true,guess in zip(true_predictions,model_predictions):

    if(guess == label_map[true]):
        correct_sum+=1
    else:
        wrong_predictions.append(true)


success_rate = correct_sum/len(true_predictions)
print(f"num samples predicted: {len(true_predictions)}")
print(f"num samples predicted correctly: {correct_sum}")
print(f"success rate = {success_rate*100}%")
print(wrong_predictions)



def plot_incorrect_predictions(wrong_predictions):

    for image_file in os.listdir(test_lines_directory):
      filename = os.path.splitext(os.path.basename(image_file))[0]
      if  filename in wrong_predictions:
        img = Image.open(test_line_directory+'/'+image_file)
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Test line {filename}")
        plt.show()

    plt.show()

def display_confusion_matrix(true_labels, predicted_labels, class_names):
    cm = confusion_matrix(true_labels, predicted_labels)

    # Set the color scheme
    cmap = plt.cm.binary  # Use binary colormap for black and white

    # Create the ConfusionMatrixDisplay with the custom color scheme
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names, cmap=cmap)

    # Plot the confusion matrix
    disp.plot(values_format='d')

    # Customize the background color
    plt.gca().set_facecolor('white')

    # Customize the color of the dots (black)
    plt.scatter([], [], c='black', label='True Positives')
    plt.scatter([], [], c='white', edgecolors='black', linewidth=1, label='False Negatives')
    plt.scatter([], [], c='white', edgecolors='black', linewidth=1, marker='X', label='False Positives')
    plt.scatter([], [], c='white', edgecolors='black', linewidth=1, marker='s', label='True Negatives')

    # Set the number of tick locations based on the number of classes
    num_classes = len(class_names)
    plt.xticks(np.arange(num_classes), class_names, rotation=45)

    # Show the plot
    plt.legend()
    plt.show()



true_predictions = [str(label) for label in true_predictions]
model_predictions = [str(label) for label in model_predictions]
display_confusion_matrix(true_predictions, model_predictions, list(label_map.keys()))

plot_incorrect_predictions(true_predictions,label_map, model_predictions, test_line_directory)















