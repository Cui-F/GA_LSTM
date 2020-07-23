import tensorflow as tf
import numpy as np

import os

from data_process import csv_to_dataset


tf.keras.backend.set_floatx('float32')

class Attention(tf.keras.layers.Layer):
  def __init__(self, units_num):
    super(Attention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units_num)
    self.W2 = tf.keras.layers.Dense(units_num)
    self.W3 = tf.keras.layers.Dense(units_num)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):

    hidden_with_time_axis_h = tf.expand_dims(query[0], 1)
    hidden_with_time_axis_c = tf.expand_dims(query[1], 1)

    score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis_h)+self.W3(hidden_with_time_axis_c)))

    attention_weights = tf.nn.softmax(score, axis=1)

    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class Encoder(tf.keras.Model):
    def __init__(self, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units

        #可以考虑通过ga调整的参数  recurrent_initializer 有多个值（orthogonal、glorot_uniform）
        self.lstm = tf.keras.layers.LSTM(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):

        self.lstm.states[0] = hidden[0]
        self.lstm.states[1] = hidden[1]

        output, state_h, state_c = self.lstm(x)

        return output, state_h, state_c

    def initialize_hidden_state(self):
        return tf.zeros([self.batch_sz, self.enc_units])

class Decoder(tf.keras.Model):
  def __init__(self, result_size, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.lstm = tf.keras.layers.LSTM(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

    #目前对只对客流进行预测，
    self.fc = tf.keras.layers.Dense(result_size)

    # 通过Attention对第1次编码输出加权
    self.attention = Attention(self.dec_units)

  def call(self,hidden, enc_output):
    # 编码器输出 （enc_output） 的形状 == （批大小，隐藏层大小）
    context_vector, attention_weights = self.attention(hidden, enc_output)

    context_vector_ = tf.expand_dims(context_vector, 1)

    output, state, _ = self.lstm(context_vector_)

    # 输出的形状 == （批大小 * 1，隐藏层大小）
    output = tf.reshape(output, (-1, output.shape[2]))

    # 输出的形状 == （批大小，vocab）
    x = self.fc(output)

    return x

@tf.function
def train_step(input, targ, enc_hidden):

    with tf.GradientTape() as tape:
        enc_output, enc_hidden_h,  enc_hidden_c= encoder(input, enc_hidden)
        enc_hidden = [enc_hidden_h,  enc_hidden_c]

        predictions = decoder(enc_hidden, enc_output)

        loss = tf.keras.losses.MSE(targ, predictions)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    loss_ = tf.reduce_sum(loss)

    return loss_

#遗传算法部分

def cal_pop_fitness(pop):

    fitness =[]
    for p in pop:

        encoder = Encoder(enc_units=units_num, batch_sz=batch_szie)
        init_hidden = encoder.initialize_hidden_state()

        decoder = Decoder(result_size, units_num, batch_sz=batch_szie)

        steps_per_epoch = input.shape[0] // batch_szie

        for epoch in range(EPOCHS):

            enc_hidden = [init_hidden, init_hidden]
            total_loss = 0

            for (batch, (inp, targ)) in enumerate(dataset_.take(steps_per_epoch)):
                batch_loss = train_step(inp, targ, enc_hidden)
                total_loss += batch_loss

            fitness.append(-total_loss)

    return np.array(fitness)

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = np.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover, num_mutations=1):
    mutations_counter = np.uint8(offspring_crossover.shape[1] / num_mutations)
    # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            # The random value to be added to the gene.
            random_value = np.random.uniform(-1.0, 1.0, 1)
            offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
            gene_idx = gene_idx + mutations_counter
    return offspring_crossover

units_num = 32
batch_szie = 16
time_step = 16
attention_units = 16
step = 5


BUFFER_SIZE = 10000
result_size = 1

EPOCHS = 1000

X, Y = csv_to_dataset(r'Metro_Interstate_Traffic_Volume.csv')

input = np.array([X[step_count*step: step_count*step+time_step] for step_count in range((X.shape[0]-time_step-1)//step)])
targ = np.array([Y[step_count*step+time_step+1] for step_count in range((X.shape[0]-time_step-1)//step)])

#mse0 = 0.01
input_ = input[:int(input.shape[0]*0.8)]
targ_ = targ[:int(targ.shape[0]*0.8)]


#mse1 = 0.2
test_input = input[int(input.shape[0]*0.8):]
test_targ = targ[int(input.shape[0]*0.8):]


dataset = tf.data.Dataset.from_tensor_slices((input_, targ_)).shuffle(BUFFER_SIZE)

dataset_ = dataset.batch(batch_szie, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset_))

encoder = Encoder(enc_units=units_num, batch_sz=batch_szie)

optimizer = tf.keras.optimizers.Adam()

init_hidden = encoder.initialize_hidden_state()

decoder = Decoder(result_size, units_num, batch_sz=batch_szie)

steps_per_epoch = input.shape[0] // batch_szie

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)



for epoch in range(EPOCHS):

    enc_hidden = [init_hidden, init_hidden]
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset_.take(steps_per_epoch)):

        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

    print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss/steps_per_epoch))

    if epoch % 200 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)



def get_model(layers=2, units=8):

    input_ = tf.keras.layers.Input(shape=())

    for i in  range(layers):
        enc


    model = tf.keras.Model()
    return model


def get_data(batch_size=32, time_step = 16):

    X, Y = csv_to_dataset(r'Metro_Interstate_Traffic_Volume.csv')

    input = np.array([X[step_count*step: step_count*step+time_step] for step_count in range((X.shape[0]-time_step-1)//step)])
    targ = np.array([Y[step_count*step+time_step+1] for step_count in range((X.shape[0]-time_step-1)//step)])

    #训练集
    input_ = input[:int(input.shape[0]*0.8)]
    targ_ = targ[:int(targ.shape[0]*0.8)]

    #测试集
    test_input = input[int(input.shape[0]*0.8):]
    test_targ = targ[int(input.shape[0]*0.8):]

    dataset = tf.data.Dataset.from_tensor_slices((input_, targ_)).shuffle(BUFFER_SIZE)

    dataset_ = dataset.batch(batch_szie, drop_remainder=True)

    return dataset_, test_input, test_targ
