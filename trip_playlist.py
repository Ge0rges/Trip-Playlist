import spotipy
from spotipy.oauth2 import SpotifyOAuth
from collections import defaultdict, Counter
import numpy as np
from spotipy.client import SpotifyException
import math
import json
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from sklearn.preprocessing import MultiLabelBinarizer

# Spotify credentials
client_id = ""
client_secret = ""
redirect_uri = "http://localhost/"


# Authentication
scope = (
    "user-library-read user-read-recently-played playlist-modify-public user-top-read"
)
sp = spotipy.Spotify(
    auth_manager=SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=scope,
    )
)


def get_user_library():
    results = sp.current_user_saved_tracks(limit=50)
    tracks = results["items"]
    while results["next"]:
        results = sp.next(results)
        tracks.extend(results["items"])
    return tracks


def get_recently_played():
    results = sp.current_user_recently_played(limit=50)
    tracks = results["items"]
    while results["next"]:
        results = sp.next(results)
        tracks.extend(results["items"])
    return [item["track"] for item in tracks]


def count_plays(recently_played):
    play_count = Counter()
    for track in recently_played:
        play_count[track["id"]] += 1
    return play_count


def get_track_popularity(tracks):
    popularity_dict = {}
    for track in tracks:
        popularity_dict[track["id"]] = track["popularity"]
    return popularity_dict


def get_user_top_tracks(limit=50):
    """
    Retrieve the user's top library_tracks.

    :param time_range: The time range for the top library_tracks. Possible values: 'short_term', 'medium_term', 'long_term'.
    :param limit: The maximum number of library_tracks to retrieve.
    :return: List of top library_tracks.
    """
    results = sp.current_user_top_tracks(time_range="long_term", limit=limit)["items"]
    results += sp.current_user_top_tracks(time_range="short_term", limit=limit)["items"]
    results += sp.current_user_top_tracks(time_range="medium_term", limit=limit)["items"]

    return results


def get_audio_features(tracks):
    results = {}
    chunk_size = 100

    for i in range(0, len(tracks), chunk_size):
        track_chunk = tracks[i: i + chunk_size]
        track_ids = [track["id"] for track in track_chunk]

        try:
            audio_features = sp.audio_features(track_ids)
            for idx, features in enumerate(audio_features):
                if features is not None:
                    results[track_chunk[idx]["id"]] = features

        except SpotifyException as e:
            print(f"Error getting audio features: {e.headers}")
            for track in track_ids:
                results[track] = None

        # Sleep thread for 30 seconds
        print(f"Processed {i + chunk_size} library_tracks...")

    return results


def get_artist_genres(tracks):
    artist_ids = {track["track"]["artists"][0]["id"] for track in tracks}

    # Split artist IDs into batches of 100
    batch_size = 50
    artist_id_batches = [
        list(artist_ids)[i: i + batch_size]
        for i in range(0, len(artist_ids), batch_size)
    ]

    # Create a dictionary to map artist IDs to genres
    artist_genres = {}
    for batch in artist_id_batches:
        artists_info = sp.artists(batch)
        for artist in artists_info["artists"]:
            artist_genres[artist["id"]] = artist["genres"]

    return artist_genres


def group_by_genre(library_tracks, play_count, popularity_dict, audio_features, top_tracks, artist_genres):
    genre_dict = defaultdict(list)

    # Iterate over library_tracks and group by genre
    for item in library_tracks:
        track = item["track"]
        artist_id = track["artists"][0]["id"]
        genres = artist_genres.get(artist_id, [])

        for genre in genres:
            score = play_count[track["id"]] * 30 + popularity_dict[track["id"]]
            if track["id"] in top_tracks:
                score = np.inf
            audio_features[track["id"]]["genres"] = genres
            genre_dict[genre].append((track, score, audio_features[track["id"]]))

    # Remove genre that are too niche
    keys_to_remove = []
    for key, value in genre_dict.items():
        if len(value) < 5:
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del genre_dict[key]

    return genre_dict


def get_top_songs_by_genre(genre_dict, top_n=3):
    songs_added = []
    top_songs = {}

    genre_dict = dict(sorted(genre_dict.items(), key=lambda item: len(item[1])))

    for genre, tracks in genre_dict.items():
        # Remove last 3 genre from tracks
        sorted_tracks = sorted(tracks, key=lambda x: x[1], reverse=True)  # Sort by score
        genre_top_tracks = []

        i = 0
        for track in sorted_tracks:
            if track[0]["id"] not in songs_added and track[0]["name"] not in songs_added:
                songs_added.append(track[0]["id"])
                songs_added.append(track[0]["name"])
                genre_top_tracks.append(track)
                i += 1

            if i == top_n:
                break

        if len(genre_top_tracks) == 0:
            print(f"No tracks found for genre:{genre}, despite starting with {len(tracks)} tracks")

        else:
            top_songs[genre] = genre_top_tracks

    return top_songs


def order_songs_by_genre_order(top_songs_by_genre, artist_genres):
    genre_lists = artist_genres.values()
    pair_counts = defaultdict(int)

    # Generate all sorted_pairs of genres in each list
    for genre_list in genre_lists:
        genre_list = genre_list[:4]
        for pair in combinations(genre_list, 2):
            sorted_pair = tuple(sorted(pair))  # Sort to avoid ('rock', 'pop') and ('pop', 'rock') being different
            pair_counts[sorted_pair] += 1

        if len(genre_list) == 1:
            pair_counts[(genre_list[0], genre_list[0])] += 1

    # Sort sorted_pairs by frequency in descending order
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])

    # Create a sequence of genres
    sequence = list(sorted_pairs[0][0])

    for pair in sorted_pairs[1:]:
        added = False

        for i, genre in enumerate(sequence):
            if pair[0][0] == genre:
                if pair[0][1] not in sequence:
                    j = i + 1
                    while j < len(sequence) and (genre, sequence[j]) in sorted_pairs:
                        j += 1
                    sequence.insert(j, pair[0][1])
                    added = True
                    break

            if pair[0][1] == genre:
                if pair[0][0] not in sequence:
                    j = i + 1
                    while j < len(sequence) and (genre, sequence[j]) in sorted_pairs:
                        j += 1
                    sequence.insert(j, pair[0][0])
                    added = True
                    break

        if not added:
            if pair[0][0] not in sequence and pair[0][1] not in sequence:
                sequence.append(pair[0][0])
                if pair[0][0] != pair[0][1]:
                    sequence.append(pair[0][1])


    # for genre in top_songs_by_genre.keys():
    #     if genre not in sequence:
    #         assert False, f"Genre {genre} not in sequence"

    # Add the songs following the sequence
    ordered_songs = []
    for genre in sequence:
        if genre in top_songs_by_genre.keys():
            ordered_songs.extend(top_songs_by_genre[genre])

    # Do a rolling window sort over song audio features
    features = np.array([[song[2]["tempo"], song[2]["key"], song[2]["danceability"]] for song in ordered_songs])
    from numpy.lib.stride_tricks import sliding_window_view

    window_length = 9
    windowed_features = sliding_window_view(np.array(features), window_shape=(window_length, features.shape[1]))

    sorted_songs_per_window = []

    # Iterate through each window
    for i, window in enumerate(windowed_features):
        # Create a list of tuples (feature_row, song) for sorting
        song_feature_pairs = list(zip(window[0, :], ordered_songs[i*window_length: (i+1)*window_length]))

        # Sort based on the feature values
        song_feature_pairs.sort(key=lambda x: tuple(x[0]))  # Sorting by feature values

        # Extract the sorted songs
        sorted_songs_per_window.extend([song[0] for _, song in song_feature_pairs])

    return sorted_songs_per_window


def order_songs_by_genre_features(top_songs_by_genre, song_info_by_genre):
    features = []
    songs = []
    for genre in top_songs_by_genre.keys():
        songs_audio_features = [song[2] for song in top_songs_by_genre[genre]]
        songs_in_genre = [song[0] for song in top_songs_by_genre[genre]]

        # Get features for each song in the genre
        for feature, song in zip(songs_audio_features, songs_in_genre):
            features.append([feature["tempo"], feature["key"], feature["danceability"], feature["energy"], feature["valence"], feature["genres"][-1]])
            songs.append(song)

    # Normalize the features
    all_genres = [entry[-1] for entry in features]

    # Use MultiLabelBinarizer to one-hot encode the list of genres
    mlb = MultiLabelBinarizer()
    one_hot_encoded_genres = mlb.fit_transform(all_genres)

    # Combine one-hot encoded genres with the rest of the features
    for i, feat in enumerate(features):
        # Append other features followed by the one-hot encoded genres
        features[i] = features[i][:-1] + list(one_hot_encoded_genres[i])

    # Normalize the features
    scaler = MinMaxScaler()
    normalized_songs = scaler.fit_transform(features)

    # Calculate pairwise Euclidean distances
    distance_matrix = squareform(pdist(normalized_songs, metric='euclidean'))

    # Create a graph from the distance matrix
    G = nx.Graph()
    num_songs = len(distance_matrix)

    for i in range(num_songs):
        for j in range(i + 1, num_songs):
            G.add_edge(i, j, weight=distance_matrix[i, j])

    # Use the approximation algorithm for TSP
    tsp_path = nx.approximation.traveling_salesman_problem(G, cycle=False)

    # Print the order of songs
    print("Order of songs for optimal flow:")
    print(tsp_path)

    # Create a list of songs in the optimal order
    final_songs = [songs[i] for i in tsp_path]

    return final_songs


def create_or_update_playlist(sp, user_id, playlist_name, tracks):
    playlists = sp.user_playlists(user_id)
    playlist_id = None

    for playlist in playlists["items"]:
        if playlist["name"] == playlist_name:
            playlist_id = playlist["id"]
            break

    if playlist_id:
        track_uris = [track["uri"] for track in tracks]
        num_batches = math.ceil(len(track_uris) / 50)

        sp.playlist_replace_items(playlist_id, track_uris[0:50])

        for i in range(1, num_batches):
            batch = track_uris[i * 50: (i + 1) * 50]
            sp.playlist_add_items(playlist_id, batch)

    else:
        playlist_id = sp.user_playlist_create(user_id, playlist_name)["id"]

        track_uris = [track["uri"] for track in tracks]
        num_batches = math.ceil(len(track_uris) / 50)
        for i in range(num_batches):
            batch = track_uris[i * 50: (i + 1) * 50]
            sp.playlist_add_items(playlist_id, batch)


def main():
    # Check if the JSON file exists
    try:
        with open("all_data.json", "r") as f:
            all_data = json.load(f)
            user_id = all_data["user_id"]
            library_tracks = all_data["library_tracks"]
            popularity_dict = all_data["popularity_dict"]
            top_tracks = all_data["top_tracks"]
            audio_features_dict = all_data["audio_features_dict"]
            artist_genres = all_data["artist_genres"]
            recently_played = all_data["recently_played"]

    except FileNotFoundError:
        print("Getting user ID...")
        user_id = sp.current_user()["id"]
        print("Getting library...")
        library_tracks = get_user_library()
        print("Getting recently played...")
        recently_played = get_recently_played()
        print("Getting popularity...")
        popularity_dict = get_track_popularity([item["track"] for item in library_tracks])
        print("Getting top songs...")
        top_tracks = get_user_top_tracks()
        print("Getting audio features...")
        audio_features_dict = get_audio_features([item["track"] for item in library_tracks])
        print("Getting artist genres...")
        artist_genres = get_artist_genres(library_tracks)

        # Write everything to a JSON file
        with open("all_data.json", "w") as f:
            all_data = {"user_id": user_id,
                        "library_tracks": library_tracks,
                        "popularity_dict": popularity_dict,
                        "recently_played": recently_played,
                        "top_tracks": top_tracks,
                        "audio_features_dict": audio_features_dict,
                        "artist_genres": artist_genres}
            json.dump(all_data, f)

    print("Getting counts...")
    play_count = count_plays(recently_played)
    print("Grouping track features...")
    song_info_by_genre = group_by_genre(library_tracks, play_count, popularity_dict, audio_features_dict, top_tracks,
                                        artist_genres)
    print("Getting top songs by genre...")
    top_songs_by_genre = get_top_songs_by_genre(song_info_by_genre)
    print("Making order by genre...")
    final_songs_ordered_by_genre = order_songs_by_genre_order(top_songs_by_genre, artist_genres)
    print("Making order by key and tempo...")
    final_songs_ordered_by_features = order_songs_by_genre_features(top_songs_by_genre, song_info_by_genre)
    print("Creating playlists...")
    create_or_update_playlist(sp, user_id, "Trip by genre", final_songs_ordered_by_genre)
    create_or_update_playlist(sp, user_id, "Trip by tempo", final_songs_ordered_by_features)

    print("Playlist updated successfully!")


if __name__ == "__main__":
    main()
