import pandas as pd



imdb_df = pd.read_csv('/Users/ludwigvoss/PyCharmMiscProject/00_input_data/imdb.csv',encoding='utf-8')
rt_df = pd.read_csv('/Users/ludwigvoss/PyCharmMiscProject/00_input_data/rotten_tomatoes.csv',encoding='utf-8')

# === Ground Truth ===
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

def jaccard_similarity_for_columns(col1, col2):

    #sets jeders spalte werden miteinander verglichen
    # entferne Nan-Werte und alles als string
    set1 = set(col1.dropna().astype(str))
    set2 = set(col2.dropna().astype(str))
    if not set1 or not set2:
        return 0

    #jaccard wieder am start
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return intersection / union if union != 0 else 0

def instance_based_matching(df1, df2):
    matches = []

    for c1 in df1.columns:
        best_match = None
        best_score = 0

        #compare header c1 from imdb with header c2 from rt
        for c2 in df2.columns:

            score = jaccard_similarity_for_columns(df1[c1], df2[c2])

            if score > best_score:
                best_score = score
                best_match = c2
        matches.append((c1, best_match, round(best_score, 3)))

    # Ergebnisse nach Score sortieren (höchster zuerst)
    #matches.sort(key=lambda x: x[2], reverse=True)
    return matches


if __name__ == '__main__':
    res2 = []

    best_matches = instance_based_matching(imdb_df, rt_df)
    for imdb_col, rt_col, score in best_matches:
        print(f"(IMDb: {imdb_col}, R_T: {rt_col}) → Score: {score}")
        if score > 0:
            res2.append((f"Imdb.{imdb_col}", f"Rt.{rt_col}"))

    #3. Precision & Recall
    predicted_set = set(res2)
    ground_truth_set = set(G)

    true_positives = predicted_set.intersection(ground_truth_set)

    #Precision = Anteil der richtigen Vorhersagen, an allen Vorhersagen
    #Recall = Anteil der richtigen Vorhersagen, an allen tatsächlich richtigen matches
    precision = len(true_positives) / len(predicted_set) if predicted_set else 0
    recall = len(true_positives) / len(ground_truth_set) if ground_truth_set else 0

    print(f"\n4. Precision & Recall\nPrecision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")


