![alt tag](http://karpathy.github.io/assets/rnn/charseq.jpeg)
* Exemple de Recurrent Neural Network


# A voir et à savoir : 

#### Les réseaux de neurones récurrents – RNNs – ont été introduit en machine learning afin de pouvoir traiter des données séquentielles. Ils sont utilisés en reconnaissance automatique de la parole ou de l'écriture manuscrite - plus en général en reconnaissance de formes - ou encore en traduction automatique.
* Build a Recurrent Neural Net in 5 Min : https://youtu.be/cdLUzrjnlr4 (by Siraj Raval)
* Stanford Lecture "Recurrent Neural Networks, Image Captioning, LSTM" : https://youtu.be/yCC09vCHzF8 (by 
Andrej Karpathy)



#### MNIST data reshape [batch_size, n_steps, n_inputs] est attendu par le RNN 
```
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])
````

#### BasicRNNCell est une usine qui fabrique des cellules RNN (une pour chacun des "time step")
```
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
```
#### La fonction dynamic_rnn()accept en input X (shape [None, n_steps, n_inputs], comme outputs (shape [None, n_steps, n_neurons]. The states,un tensor qui contient l'état final de chacune des cellules. 

```
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

```










