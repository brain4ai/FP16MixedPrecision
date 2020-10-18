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

#-Creamos el optimizador y lo pasamos por la funcion que nos permitirá
#computar en precision mixta
#-Observamos que se especifica loss_scale como "dynamic", esto facilita
#no encontrarnos con underflow y overflow
#-Si entrenamos con la funcion .fit() el loss_scale estará por defecto
#en dynamic
optimizer = keras.optimizers.RMSprop()
optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')

#Definimos como siempre el objeto loss y nuestros train y test datasets
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
train_dataset = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                 .shuffle(10000).batch(8192))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(8192)

'''A continuación definimos el train_step, un paso de entrenamiento.
Observaremos como diferencia las funciones get_scaled_loss(loss)
y get_unscaled_gradients(gradients). Estas funciones deben usarse así
y en este orden. Se utilizan para evitar underflow en los gradientes.'''
@tf.function
def train_step(x, y):
  with tf.GradientTape() as tape:
    predictions = model(x)
    loss = loss_object(y, predictions)
    scaled_loss = optimizer.get_scaled_loss(loss)
  scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
  gradients = optimizer.get_unscaled_gradients(scaled_gradients)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss

#Ahora definimos nuestra funcion para testear resultados test_step
@tf.function
def test_step(x):
  return model(x, training=False)

#Por último creamos nuestro custom training loop
#Vamos a entrenar durante 10 epochs
for epoch in range(10):
  epoch_loss_avg = tf.keras.metrics.Mean()
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='test_accuracy')
  for x, y in train_dataset:
    loss = train_step(x, y)
    epoch_loss_avg(loss)
  for x, y in test_dataset:
    predictions = test_step(x)
    test_accuracy.update_state(y, predictions)
  print('Epoch {}: loss={}, test accuracy={}'.format(epoch, epoch_loss_avg.result(), test_accuracy.result()))