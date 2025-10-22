import unittest

from Task1 import (
    input_reader,
    orderSets,
    jaccard_similarity_algorithm,
    join_best_jaccard_matches,
    G
)

class Task1_Tests(unittest.TestCase):

    def setUp(self):
        # Setze Beispiel-Daten (anstatt CSVs zu laden)
        self.cols_imdb = ["Name", "YearRange", "ReleaseDate", "Director", "Creator",
                          "Cast", "Duration", "RatingValue", "Genre", "Description"]

        self.cols_rt = ["Name", "Year", "Release Date", "Director", "Creator",
                        "Cast", "Duration", "RatingValue", "Genre", "Description"]

    def test_jaccard_similarity_algorithm(self):
        # 100% Gleichheit
        score = jaccard_similarity_algorithm("Director", "Director")
        self.assertEqual(score, 1.0)

        # Teilweise Überschneidung
        score = jaccard_similarity_algorithm("ReleaseDate", "Release Date")
        self.assertGreater(score, 0.0)

        # Keine Ähnlichkeit
        score = jaccard_similarity_algorithm("Genre", "NothingSimilar")
        self.assertLess(score, 0.5)

    def test_join_best_jaccard_matches(self):
        # Berechne beste Matches
        best_matches = join_best_jaccard_matches(self.cols_imdb, self.cols_rt)
        result_pairs = [(f"Imdb.{a}", f"Rt.{b}") for a, b, _ in best_matches if _ > 0.5]

        # Prüfe, dass mindestens 8-10 gute Matches gefunden wurden
        self.assertGreaterEqual(len(result_pairs), 8)

    def test_against_ground_truth(self):
        # Volles Pipeline-Verhalten testen
        best_matches = join_best_jaccard_matches(self.cols_imdb, self.cols_rt)
        res1 = [(f"Imdb.{a}", f"Rt.{b}") for a, b, s in best_matches if s > 0.5]

        # Sortiere beide
        res1_sorted = sorted(res1)
        G_sorted = sorted(G)

        # Vergleiche mit Ground Truth
        self.assertEqual(res1_sorted, G_sorted, "Calculated result does not match Ground Truth!")

    if __name__ == "__main__":
        unittest.main()

