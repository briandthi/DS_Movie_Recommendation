import pandas as pd


def create_affinity_features_optimized(df):
    total_rows = len(df)

    director_count = df.groupby(["userId", "director"]).size()
    actor1_count = df.groupby(["userId", "actor1"]).size()
    actor2_count = df.groupby(["userId", "actor2"]).size()
    writer_count = df.groupby(["userId", "writer"]).size()

    df["director_count"] = df.apply(
        lambda x: director_count.get((x["userId"], x["director"]), 0), axis=1
    )
    df["actor1_count"] = df.apply(
        lambda x: actor1_count.get((x["userId"], x["actor1"]), 0), axis=1
    )
    df["actor2_count"] = df.apply(
        lambda x: actor2_count.get((x["userId"], x["actor2"]), 0), axis=1
    )
    df["writer_count"] = df.apply(
        lambda x: writer_count.get((x["userId"], x["writer"]), 0), axis=1
    )

    cols = {
        "director": "directorAvgRating",
        "actor1": "actor1AvgRating",
        "actor2": "actor2AvgRating",
        "writer": "writerAvgRating",
    }

    for person_type, avg_col in cols.items():
        df[f"{person_type}_affinity"] = 0

        for idx, row in df.iterrows():
            if row[f"{person_type}_count"] > 1:
                other_movies = df[
                    (df["userId"] == row["userId"])
                    & (df[person_type] == row[person_type])
                    & (df.index != idx)
                ]

                if len(other_movies) > 0:
                    user_person_mean = other_movies["rating"].mean()
                    df.at[idx, f"{person_type}_affinity"] = (
                        user_person_mean - row[avg_col]
                    )

    return df
