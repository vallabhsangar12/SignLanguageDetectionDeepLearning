from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential

# CNN Model Definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(150, activation='relu'),
    Dropout(0.25),
    Dense(6, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=12,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.15,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_dir = r"D:\MS-Edunet Foundation Internship\SignLanguageDetectionDeepLearningProject-main-P4\HandGestureDataset\train"
val_dir = r"D:\MS-Edunet Foundation Internship\SignLanguageDetectionDeepLearningProject-main-P4\HandGestureDataset\test"

training_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    color_mode='grayscale',
    batch_size=8,
    classes=['NONE', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'],
    class_mode='categorical'
)

val_set = val_datagen.flow_from_directory(
    val_dir,
    target_size=(256, 256),
    color_mode='grayscale',
    batch_size=8,
    classes=['NONE', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'],
    class_mode='categorical'
)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10),
    ModelCheckpoint('sign_language_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
]

# Training
model.fit(
    training_set,
    steps_per_epoch=len(training_set),
    epochs=25,
    validation_data=val_set,
    validation_steps=len(val_set),
    callbacks=callbacks
)
