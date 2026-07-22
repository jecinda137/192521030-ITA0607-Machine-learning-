import pandas as pd

# Movie Dataset
movies = {
    "Movie": [
        "Avengers: Endgame",
        "The Dark Knight",
        "Inception",
        "Titanic",
        "Frozen",
        "Toy Story",
        "Interstellar",
        "Joker",
        "The Conjuring",
        "Finding Nemo"
    ],
    "Genre": [
        "Action",
        "Action",
        "Sci-Fi",
        "Romance",
        "Animation",
        "Animation",
        "Sci-Fi",
        "Drama",
        "Horror",
        "Animation"
    ],
    "Rating": [
        8.4,
        9.0,
        8.8,
        7.9,
        7.4,
        8.3,
        8.7,
        8.4,
        7.5,
        8.2
    ]
}

# Create DataFrame
df = pd.DataFrame(movies)

# Display all movies
print("Available Movies:\n")
print(df)

# User input
genre = input("\nEnter your preferred genre: ")
min_rating = float(input("Enter minimum rating (1-10): "))

# Recommend movies
recommend = df[
    (df["Genre"].str.lower() == genre.lower()) &
    (df["Rating"] >= min_rating)
]

# Display recommendations
print("\nRecommended Movies:\n")

if not recommend.empty:
    print(recommend[["Movie", "Genre", "Rating"]])
else:
    print("No movies found matching your preferences.")
