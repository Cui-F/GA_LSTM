import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error

from data_process import csv_to_dataset


#遗传算法部分
#调整模型参数
#

def cal_pop_fitness(population):
    """

    :param pop:
    :return:
    """

    fitness =[]
    for p in population:
        batch_size, time_step, step, units = p

        dataset, test_input, test_targ = get_data(batch_size=batch_size, time_step=time_step, step=step)
        model = get_model(units=units, time_step=time_step)
        model.summary()

        model.fit(dataset, epochs=1)

        prec = model.predict(test_input)

        mse = mean_squared_error(test_targ, prec);

        fitness.append(mse)

        tf.keras.backend.clear_session()

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
    mutation_range = {0: 60, 1: 60, 2: 12, 3: 500}
    mutation_min = {0: 8, 1: 8, 2: 8, 3: 64}
    mutation_max = {0: 128, 1: 128, 2: 32, 3: 1024}

    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            # The random value to be added to the gene.
            random_value = np.random.uniform(-1.0, 1.0, 1)*mutation_range[gene_idx]
            offspring_crossover[idx, gene_idx] = np.clip(offspring_crossover[idx, gene_idx] + random_value, mutation_min[gene_idx], mutation_max[gene_idx])
            gene_idx = gene_idx + mutations_counter
    return offspring_crossover


#lstm模型部分
#

def get_model(units=8, time_step=32):
    """
    :param units:
    :param time_step:
    :return:
    """

    input_ = tf.keras.layers.Input(shape=(time_step,10))
    #使用1层lstm作为编码器
    enc_output = tf.keras.layers.GRU(units, recurrent_initializer='glorot_uniform')(input_)

    dec_output = tf.keras.layers.GRU(units, recurrent_initializer='glorot_uniform')(enc_output)

    result = tf.keras.layers.Dense(1)(dec_output)
    result_ = tf.keras.layers.Reshape((1,))(result)

    model = tf.keras.Model(input_, result_)

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mse',])

    return model


def get_data(batch_size=32, time_step = 32, step=5):
    """
    :param batch_size:
    :param time_step:
    :param step:
    :return:
    """

    X, Y = csv_to_dataset(r'Metro_Interstate_Traffic_Volume.csv')

    input = np.array([X[step_count*step: step_count*step+time_step] for step_count in range((X.shape[0]-time_step-1)//step)])
    targ = np.array([Y[step_count*step+time_step+1] for step_count in range((X.shape[0]-time_step-1)//step)])

    #训练集
    input_ = input[:int(input.shape[0]*0.8)]
    targ_ = targ[:int(targ.shape[0]*0.8)]

    #测试集,用于计算fitness
    test_input = input[int(input.shape[0]*0.8):]
    test_targ = targ[int(input.shape[0]*0.8):]

    dataset = tf.data.Dataset.from_tensor_slices((input_, targ_)).shuffle(BUFFER_SIZE)

    dataset_ = dataset.batch(batch_size, drop_remainder=True)

    return dataset_, test_input, test_targ


def main():

    #population =[[batch_size, time_step, step, units], ...]

    batch_size = np.random.randint(low=8, high=128, size=(8,))
    time_step = np.random.randint(low=8, high=128, size=(8,))
    step = np.random.randint(low=8, high=32, size=(8,))
    units = np.random.randint(low=64, high=1024, size=(8,))

    population = np.array(list(zip(batch_size, time_step, step, units)), dtype=np.int)

    num_generations = 1000

    for generation in range(num_generations):
        print("Generation {}".format(generation))

        fitness = cal_pop_fitness(population)

        parents = select_mating_pool(population, fitness, num_parents_mating)

        offspring_crossover = crossover(parents,
                                        offspring_size=(pop_size[0] - parents.shape[0], 4))

        offspring_mutation = mutation(offspring_crossover, num_mutations=2)

        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = offspring_mutation

        print(population)

    fitness = cal_pop_fitness(population)
    index = np.argmax(fitness)
    best_model_parm = population[index]
    batch_size, time_step, step, units = best_model_parm

    model = get_model(units, time_step)
    model.summary()




if __name__ == "__main__":

    BUFFER_SIZE = 10000
    sol_per_pop = 8
    num_parents_mating = 4
    pop_size = (sol_per_pop, 4)

    main()