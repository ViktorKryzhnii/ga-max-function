import random

from deap import base, creator, tools

NUM_GENES = 3
INIT_LOW = -10.0
INIT_UP = 10.0

POPULATION_SIZE = 100
NUM_GENERATIONS = 100
PROB_CROSSING = 0.9
PROB_MUTATING = 0.2
MUTATION_MU = 0.0
MUTATION_SIGMA = 0.1
TOURNAMENT_SIZE = 3
ELITE_SIZE = 2
RANDOM_STATE = 7


def eval_func(individual):
    x, y, z = individual
    value = 1 / (1 + (x - 2) ** 2 + (y + 1) ** 2 + (z - 1) ** 2)
    return (value,)


def create_toolbox():
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register('attr_float', random.uniform, INIT_LOW, INIT_UP)
    toolbox.register('individual', tools.initRepeat, creator.Individual,
                     toolbox.attr_float, NUM_GENES)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', eval_func)
    toolbox.register('mate', tools.cxOnePoint)
    toolbox.register('mutate', tools.mutGaussian, mu=MUTATION_MU,
                     sigma=MUTATION_SIGMA, indpb=1 / NUM_GENES)
    toolbox.register('select', tools.selTournament, tournsize=TOURNAMENT_SIZE)
    return toolbox


def evaluate_population(toolbox, individuals):
    for ind in individuals:
        ind.fitness.values = toolbox.evaluate(ind)


def apply_crossover(toolbox, offspring):
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < PROB_CROSSING:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values


def apply_mutation(toolbox, offspring):
    for mutant in offspring:
        if random.random() < PROB_MUTATING:
            toolbox.mutate(mutant)
            del mutant.fitness.values


def make_next_generation(toolbox, population):
    elite = list(map(toolbox.clone, tools.selBest(population, ELITE_SIZE)))
    selected = toolbox.select(population, POPULATION_SIZE - ELITE_SIZE)
    offspring = list(map(toolbox.clone, selected))

    apply_crossover(toolbox, offspring)
    apply_mutation(toolbox, offspring)

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    evaluate_population(toolbox, invalid_ind)
    return elite + offspring


def print_stats(generation, population):
    fits = [ind.fitness.values[0] for ind in population]
    mean = sum(fits) / len(fits)
    std = abs(sum(f * f for f in fits) / len(fits) - mean ** 2) ** 0.5
    print(f'Покоління {generation:3d}: min = {min(fits):.6f}, max = {max(fits):.6f}, '
          f'середнє = {mean:.6f}, ст. відхилення = {std:.6f}')


def evolve(toolbox):
    print('=== Процес еволюції ===')
    population = toolbox.population(n=POPULATION_SIZE)
    evaluate_population(toolbox, population)
    print_stats(0, population)

    for generation in range(1, NUM_GENERATIONS + 1):
        population = make_next_generation(toolbox, population)
        print_stats(generation, population)

    return tools.selBest(population, 1)[0]


def print_result(best):
    x, y, z = best
    print('\n=== Результат ===')
    print(f'Найкраща особина: x = {x:.6f}, y = {y:.6f}, z = {z:.6f}')
    print(f'Максимум функції: f = {best.fitness.values[0]:.6f}')
    print('Точний максимум: x = 2, y = -1, z = 1, f = 1')


def main():
    random.seed(RANDOM_STATE)
    toolbox = create_toolbox()
    best = evolve(toolbox)
    print_result(best)


if __name__ == '__main__':
    main()
