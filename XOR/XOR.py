

import numpy as np
import tensorflow as tf

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(2,activation="sigmoid"),
    tf.keras.layers.Dense(1,activation="sigmoid")
]
)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.optimizers.Adam(0.1)

)
model.fit(x,y,epochs=250)

test = model.predict(x)
j = 0
for i in test:
  if i > 0.5 :
    print(f"{x[j]} = 1")
  else:
    print(f"{x[j]} = 0")
  j +=1

