import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


data_dir = "./kvasir-dataset"
categories = os.listdir(data_dir)
img_size = (160, 160) 
batch_size = 64  


data_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

train_data = data_gen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = data_gen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)


base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(img_size[0], img_size[1], 3)
)


base_model.trainable = False


model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(len(categories), activation='softmax')
])


model.compile(
    optimizer=Adam(learning_rate=0.001),  
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


callbacks = [
    EarlyStopping(
        monitor='val_loss',  # Changed to monitor loss
        patience=7,  # Increased patience
        restore_best_weights=True,
        min_delta=0.001  # Reduced min delta
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]


history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    batch_size=32,  
    callbacks=callbacks,
    verbose=1
)


val_labels = val_data.classes
val_preds = model.predict(val_data)
pred_classes = np.argmax(val_preds, axis=1)

print("\nClassification Report:")
print(classification_report(val_labels, pred_classes, target_names=categories))


plt.figure(figsize=(12, 8))
conf_matrix = confusion_matrix(val_labels, pred_classes)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=categories, yticklabels=categories)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Over Time')
plt.legend()
plt.show()


def classify_image(image_path, confidence_threshold=0.85):
    """
    Classify an image and show results with confidence score
    """
    
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    
    prediction = model.predict(img_array)
    confidence = np.max(prediction) * 100
    predicted_class = categories[np.argmax(prediction)]  
    
    
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    
    if confidence/100 >= confidence_threshold:
        status = "Prediction"  
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")
    else:
        status = "Low Confidence Warning"
        print(f"Warning: Low confidence prediction ({confidence:.2f}%)")
        print(f"Predicted class: {predicted_class} - Consider getting a second opinion")
    
    plt.title(f"{status}\nClass: {predicted_class}\nConfidence: {confidence:.2f}%")
    plt.axis('off')
    plt.show()
    
    return predicted_class, confidence


result_class, result_confidence = classify_image(
    "kvasir-dataset/dyed-resection-margins/0b05f616-e6b7-4a91-b8e6-1a011561e7f7.jpg"
)
