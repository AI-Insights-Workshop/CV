from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt

# Create an instance of ImageDataGenerator with various augmentations
datagen = ImageDataGenerator(
    # Write Your Code Here
    fill_mode='nearest'
)

# Load an image
img = load_img('imgseg1.jpg')  
x = img_to_array(img)  
x = x.reshape((1,) + x.shape)  

# Generate batches of augmented images
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()