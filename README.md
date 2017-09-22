# A voir et à savoir : 

#### Les réseaux de neurones récurrents – RNNs – ont été introduit en machine learning afin de pouvoir traiter des données séquentielles. Ils sont utilisés en reconnaissance automatique de la parole ou de l'écriture manuscrite - plus en général en reconnaissance de formes - ou encore en traduction automatique.
* Build a Recurrent Neural Net in 5 Min : https://youtu.be/cdLUzrjnlr4 (by Siraj Raval)
* Stanford Lecture "Recurrent Neural Networks, Image Captioning, LSTM" : https://youtu.be/yCC09vCHzF8 (by 
Andrej Karpathy)


#### MNIST data reshape [batch_size, n_steps, n_inputs] as is expected by the RNN
```
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])
````

#### BasicRNNCell as a factory that creates copies of the cell to build the unrolled RNN (one for each time step)
```
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
```

#### The dynamic_rnn() function uses a while_loop() operation to run over the cell the appropriate number of times. It also accepts a single tensor for all inputs at every time step (shape [None, n_steps, n_inputs]) and it outputs a single tensor for all outputs at every time step (shape [None, n_steps, n_neurons]). The states tensor contains the final state of each cell

```
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

```










