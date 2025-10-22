import pandas as pd


imdb = '/Users/ludwigvoss/PyCharmMiscProject/00_input_data/imdb.csv'
rt = '/Users/ludwigvoss/PyCharmMiscProject/00_input_data/rotten_tomatoes.csv'
G = [
    ("Imdb.Name", "Rt.Name"),
    ("Imdb.YearRange", "Rt.Year"),
    ("Imdb.ReleaseDate", "Rt.Release Date"),
    ("Imdb.Director", "Rt.Director"),
    ("Imdb.Creator", "Rt.Creator"),
    ("Imdb.Cast", "Rt.Cast"),
    ("Imdb.Duration", "Rt.Duration"),
    ("Imdb.RatingValue", "Rt.RatingValue"),
    ("Imdb.Genre", "Rt.Genre"),
    ("Imdb.Description", "Rt.Description")
]
res1 = []


def input_reader(prompt: str):
    return pd.read_csv(prompt,encoding='utf-8',sep=',')

def orderSets(list1, list2) -> (list, list):
    list1 = list(list1)
    list2 = list(list2)
    list1.sort()
    list2.sort()
    return list1, list2

#https://www.geeksforgeeks.org/data-science/how-to-calculate-jaccard-similarity-in-python/
def jaccard_similarity_algorithm(list1, list2):
    set1 = set(list1)
    set2 = set(list2)

    # intersection of two sets
    intersection = len(set1.intersection(set2))
    # Unions of two sets
    union = len(set1.union(set2))

    return intersection / union


def join_best_jaccard_matches(cols1, cols2):
    matches = []

    for col1 in cols1:
        best_match = None
        best_score = 0

        for col2 in cols2:
            score = jaccard_similarity_algorithm(col1, col2)
            if score > best_score:
                best_score = score
                best_match = col2

        matches.append((col1, best_match, round(best_score, 3)))

    return matches



# Main
if __name__ == '__main__':

    #0. store input
    t1 = input_reader(imdb).columns
    t2 = input_reader(rt).columns

    #1. order list
    temp1, temp2 = orderSets(t1, t2)
    print(f'1. ordered Input as Sets: \n{temp2},\n{temp1} \n')

    #2. find best matches via jaccard
    best_matches = join_best_jaccard_matches(temp1, temp2)

    print(f'2. joining best jaccard matches: \n')

    for imdb_col, rt_col, score in best_matches:
        #only good matches into result
        if score > 0.5:
            res1.append((f"Imdb.{imdb_col}", f"Rt.{rt_col}"))
        print(f"⟨Imdb.{imdb_col}, Rt.{rt_col}⟩ → Score: {score}")


    print(f'\n3. campare Ground Truth and result:')
    print(f' calculated result: {res1}')
    print(f' Ground Truth: {sorted(G)} \n')

    #3. Precision & Recall
    predicted_set = set(res1)
    ground_truth_set = set(G)

    true_positives = predicted_set.intersection(ground_truth_set)

    #Precision = Anteil der richtigen Vorhersagen, an allen Vorhersagen
    #Recall = Anteil der richtigen Vorhersagen, an allen tatsächlich richtigen matches
    precision = len(true_positives) / len(predicted_set) if predicted_set else 0
    recall = len(true_positives) / len(ground_truth_set) if ground_truth_set else 0

    print(f"\n4. Precision & Recall\nPrecision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
