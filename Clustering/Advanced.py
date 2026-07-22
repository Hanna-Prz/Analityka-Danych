import random
base = [
    "chicken","beef","pork","fish","rice","pasta","potato","beans",
    "tomato","cheese","cream","milk","onion","carrot","pepper_veg"
]
spices = ["salt","pepper","basil","oregano","cumin","paprika","chili","garlic"]
with open("recipes.txt","w") as f:
    for _ in range(100):
        ingredients = random.sample(base, random.randint(3,6))
        spice_sel = random.sample(spices, random.randint(1,3))
        spices_marked = [s+"*" for s in spice_sel]
        f.write(",".join(ingredients+spices_marked)+"\n")
import math
import itertools
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

POP_SIZE = 50
MAX_GEN = 200000
CROSS_PROB = 0.8
MUT_PROB = 0.2
GRID_SIZE = 50

def plot_best_with_kmeans(best_solution, n_clusters):
    """
    Wykonuje k-means na punktach najlepszego rozwiązania
    i rysuje wynik.
    """
    points = np.array([[p[0], p[1]] for p in best_solution])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(points)
    centers = kmeans.cluster_centers_
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(
        points[:, 0],
        points[:, 1],
        c=labels,
        cmap="tab10",
        s=40,
        alpha=0.8
    )
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        c="black",
        s=200,
        marker="X",
        label="Centroidy"
    )
    plt.title(f"K-means (k={n_clusters}) na najlepszym rozwiązaniu")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.axis("equal")
    plt.show()


def plot_solution(solution):
    xs = [p[0] for p in solution]
    ys = [p[1] for p in solution]
    plt.figure(figsize=(8,8))
    plt.scatter(xs, ys, s=40)
    for i, (x,y) in enumerate(solution):
        plt.text(x+0.2, y+0.2, str(i), fontsize=6)
    plt.title("Osadzenie przepisów w 2D (najlepsze rozwiązanie)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.axis("equal")
    plt.show()

def load_recipes(filename):
    recipes = []
    for line in open(filename):
        items = line.strip().split(",")
        recipes.append(set(items))
    return recipes

def ingredient_distance(a, b):
    diff = a.symmetric_difference(b)
    dist = 0
    for x in diff:
        dist += 0.5 if x.endswith("*") else 1.0
    return dist

def random_individual():
    return [(random.uniform(0,GRID_SIZE), random.uniform(0,GRID_SIZE))
            for _ in range(N)]

def fitness(ind):
    score = 0
    for i,j in itertools.combinations(range(N),2):
        real = DIST_MATRIX[i][j]
        emb = math.dist(ind[i], ind[j])
        if real == 0:
            continue
        diff_pct = abs(emb-real)/real * 100
        if diff_pct <= 5:
            score += (6 - int(diff_pct)) * 10
        else:
            score -= int(diff_pct - 5) * 100
    return score

def crossover(a,b):
    child1, child2 = [], []
    for (x1,y1),(x2,y2) in zip(a,b):
        if random.random()<0.5:
            child1.append((x1,y2))
            child2.append((x2,y1))
        else:
            child1.append((x2,y1))
            child2.append((x1,y2))
    return child1, child2

def mutate(ind):
    i = random.randint(0,N-1)
    ind[i] = (random.uniform(0,GRID_SIZE), random.uniform(0,GRID_SIZE))

recipes = load_recipes("recipes.txt")
N = len(recipes)
DIST_MATRIX = [[ingredient_distance(recipes[i], recipes[j])
                for j in range(N)] for i in range(N)]
population = [random_individual() for _ in range(POP_SIZE)]
for gen in range(MAX_GEN):
    scored = [(fitness(ind), ind) for ind in population]
    scored.sort(reverse=True, key=lambda x:x[0])
    print(f"Gen {gen} | Best fitness = {scored[0][0]}")
    if scored[0][0] >= 0:
        print("Znaleziono rozwiązanie nieujemne")
        break
    new_pop = [scored[0][1], scored[1][1]]  # elity
    while len(new_pop) < POP_SIZE:
        p1 = random.choice(scored[:10])[1]
        p2 = random.choice(scored[:10])[1]
        if random.random() < CROSS_PROB:
            c1,c2 = crossover(p1,p2)
        else:
            c1,c2 = p1[:], p2[:]
        if random.random() < MUT_PROB:
            mutate(c1)
        if random.random() < MUT_PROB:
            mutate(c2)
        new_pop.extend([c1,c2])
    population = new_pop[:POP_SIZE]
best = max(population, key=fitness)
print("FINAL FITNESS:", fitness(best))
plot_solution(best)
for i in range(1,10):
    plot_best_with_kmeans(best,i)

