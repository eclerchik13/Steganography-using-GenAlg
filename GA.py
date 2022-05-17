from numpy.random import rand
import numpy as np
import random


init_conditions = 26


def get_middle_coffs(n):  # find coef in midrange
    arr_middle_coffs = [[i, n - 1 - i] for i in range(n)]
    arr_up_middle_coffs = [[i, n - 2 - i] for i in range(n - 1)]
    arr_down_middle_coffs = [[i, n - i] for i in range(1, n)]
    return arr_down_middle_coffs + arr_middle_coffs + arr_up_middle_coffs


def get_population(n_pop, arr_coord_gen):  # create population
    random.seed(0)
    random.seed(init_conditions)
    chrom = [random.sample(arr_coord_gen, 2) for _ in range(n_pop)]
    return chrom


def selection_tour(pop, fit_value, tournament_size):
    tournament_chrom = []
    tournament_scores = []
    for _ in range(tournament_size):
        rand_index_chrom = random.randint(0, len(pop)-1)
        tournament_chrom.append(pop[rand_index_chrom])
        tournament_scores.append(fit_value[rand_index_chrom])
    index_max_scores = tournament_scores.index(max(tournament_scores))
    return tournament_chrom[index_max_scores]


def crossover_one_point(chrom_1, chrom_2, p_cross):  # where chrom [ [x,y], [z,k] ]
    p = rand()
    if p > p_cross:
        return [chrom_1, chrom_2]
    else:
        return [[chrom_1[0], chrom_2[1]], [chrom_2[0], chrom_1[1]]]


def mutation(chrom, r_mut, gen):
    if rand() < r_mut:
        a = random.randint(0, 1)
        chrom[a] = gen
    return chrom


def find_best_chrom(b_score, b_chrom, fit_values, pop, n_pop):
    a = b_score
    b = b_chrom
    for chrom in range(n_pop):
        if fit_values[chrom] > a:
            a = fit_values[chrom]
            b = pop[chrom]
    return b, a


def reproduction(select_pop, p_cross, arr_gen, p_mut):
    res_ = []
    chrom_1, chrom_2 = random.sample(select_pop, 2)
    while (np.array(chrom_1[0]) == chrom_2[1]).all() or (np.array(chrom_1[1]) == chrom_2[0]).all():
        chrom_1, chrom_2 = random.sample(select_pop, 2)

    for c in crossover_one_point(chrom_1, chrom_2, p_cross):
        gen = random.choice(arr_gen)
        while (np.array(gen) == c[0]).all() or (np.array(gen) == c[1]).all():
            gen = (random.choice(arr_gen))
        c = mutation(c, p_mut, gen)
        res_.append(c)
    return np.array(res_)       # [chrom1, chrom2]


def ga_for_block(block, true_block, fitness_function, n, n_pop, n_iter, size_tour, p_cross, p_mut):
    arr_gen = get_middle_coffs(n)
    arr_pop = get_population(n_pop, arr_gen)
    best_chrom, best_score = 0, 0
    for generation in range(n_iter):
        arr_fitness_value = [fitness_function(p, block, true_block) for p in arr_pop]
        best_chrom, best_score = find_best_chrom(best_score, best_chrom, arr_fitness_value, arr_pop, n_pop)
        selected_pop = [selection_tour(arr_pop, arr_fitness_value, size_tour) for _ in range(n_pop)]
        children = list()
        for _ in range(0, n_pop, 2):
            res = reproduction(selected_pop, p_cross, arr_gen, p_mut)
            for i in range(2):
                children.append(res[i])
        arr_pop = children
        #print("Generation №", generation, "chrom", best_chrom, "f-f", best_score)
    return [best_chrom, best_score]

