from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout,BatchNormalization
from tensorflow.keras.regularizers import l2
class Model:
    def __init__(self) -> None:
        pass

    def get_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(112, 112, 3), kernel_regularizer=l2(0.01)),
            MaxPooling2D((4, 4)),
            Dropout(0.1),
            
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((4, 4)),
            Dropout(0.5),
            
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((4, 4)),
            Dropout(0.3),
            
            Flatten(),
            Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(5, activation='softmax')  # Assuming you have 10 classes
        ])

        return model