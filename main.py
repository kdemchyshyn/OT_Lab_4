import numpy as np

np.random.seed(42)

# input
max_iterations = 10 # <=20
items_num = 25 # <= 50
knapscak_size = 15
items = list()

max_item_weight = 5
max_item_value = 15

# params
population_size = 20
replace_percent = 0.5

def generateItems():
    for i in range(items_num):
        items.append((np.random.randint(0, max_item_weight), np.random.randint(0, max_item_value)))

    items.sort(key=lambda item: (-item[0], item[1]))
    print(f"Items: {items}")

def clipPopulation(population):
    for i in range(population.shape[0]):
        individual = population[i, :]

        while True:
            weights, total = evaluateFunction(individual)

            if weights <= knapscak_size:
                break

            index = np.where(individual == 1)[0][0]
            individual[index] = 0

    return population

def generatePopulation():
    population = np.random.randint(0, 2, size=(population_size, items_num))
    return sortPopulation(clipPopulation(population))

def printPopulation(population):
    for i in range(population_size):
        individual = population[i,:]
        weights, total = evaluateFunction(individual)

        id = int("".join(map(str, individual)), 2)
        print(f"{individual} -> {id}\t{weights} kg\t{total}$")

def evaluateFunction(individual):
    weights = 0
    total = 0
    for j in range(items_num):
        weights += items[j][0] * individual[j]
        total += items[j][1] * individual[j]

    return weights, total

def sortPopulation(population):
    sorted_population = sorted(population, key=lambda ind: (evaluateFunction(ind)[1], sum(ind)))
    return np.array(sorted_population[::-1])

def select(population):
    N = int(population_size * replace_percent)
    N += N % 2

    sample = np.zeros((N, items_num), dtype=int)

    probabilities = np.array([evaluateFunction(ind)[1] for ind in population], dtype=float)

    if (sum(probabilities) == 0):
        return population[:N, :]

    probabilities /= sum(probabilities)

    for i in range(N):
        selection_point = np.random.random()

        selection_sum = 0
        for index in range(population_size):
            selection_sum += probabilities[index]

            if selection_sum > selection_point:
                sample[i, :] = population[index, :].copy()
                break

    return sample

def crossover(population):
    for i in range(0, population.shape[0], 2):
        ind_1 = population[i, :].copy()
        ind_2 = population[i + 1, :].copy()

        k = np.random.randint(0, items_num)

        population[i, k:] = ind_2[k:]
        population[i + 1, k:] = ind_1[k:]

    return population

def mutate(population):
    x, y = population.shape

    x = np.random.randint(0, x)
    y = np.random.randint(0, y)

    population[x][y] = population[x][y] ^ 1

    return population

def replace(population, new_individuals):
    N = len(new_individuals)
    if N == 0:
        return population

    population[-N:] = new_individuals
    return population

def main():
    generateItems()
    current_population = generatePopulation()

    print("\nPopulation 0:")
    printPopulation(current_population)

    for i in range(max_iterations):
        selected_individuals = select(current_population)
        crossover_individuals = crossover(selected_individuals)
        mutate_individuals = mutate(crossover_individuals)

        new_individuals = clipPopulation(mutate_individuals)
        current_population = replace(current_population, new_individuals)
        current_population = sortPopulation(current_population)

        print(f"\nPopulation {i + 1}:")
        printPopulation(current_population)

    print("\nResults:")
    print(f"Total value: {evaluateFunction(current_population[0])[1]}")
    print(f"Number of items: {sum(current_population[0])}")
    print(f"Items: {[items[i] for i in range(items_num) if current_population[0][i] == 1 ]}")

if __name__ == '__main__':
    main()
