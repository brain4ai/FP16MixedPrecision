import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision import experimental as mixed_precision

#Establecemos la policy especificando float16
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

#Construimos nuestro modelo
#Crearemos un modelo sencillo unicamente con capas densas fully conected.
#Los modelos muy pequeños no se benefician tanto del uso de la precision mixta
#Crearemos un par de capas densas de 4096 elementos para las pruebas
inputs = keras.Input(shape=(784,), name='digits')
num_units = 4096
dense1 = layers.Dense(num_units, activation='relu', name='dense_1')
x = dense1(inputs)
dense2 = layers.Dense(num_units, activation='relu', name='dense_2')
x = dense2(x)

#Hay ciertas capas en los modelos que deben mantenerse en fp32 para que sean
#numericamente estables, es el caso de la softmax
#Lo indicaremos con el dtype = 'float32'
#Sobreescribir la policy poniendola a fp32 solo suele ser necesario en la ultima capa softmax
x = layers.Dense(10, name='dense_logits')(x)
outputs = layers.Activation('softmax', dtype='float32', name='predictions')(x)

#Terminamos nuestro modelo como sigue
model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

#Obtenemos y particionamos nuestro dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

#Entrenamos nuestro modelo con el método .fit
history = model.fit(x_train, y_train,
                    batch_size=8192,
                    epochs=10,
                    validation_split=0.2)

#Observamos que tal ha ido el entrenamiento
test_scores = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])
