import spotipy
from spotipy.oauth2 import SpotifyOAuth
from collections import defaultdict, Counter
import numpy as np
from spotipy.client import SpotifyException
import math
import json
from itertools import combinations

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
            genre_dict[genre].append((track, score, audio_features[track["id"]]))

    return genre_dict


def get_top_songs_by_genre(genre_dict, top_n=3):
    songs_added = []
    top_songs = {}

    genre_dict = dict(sorted(genre_dict.items(), key=lambda item: len(item[1])))

    for genre, tracks in genre_dict.items():
        sorted_tracks = sorted(tracks, key=lambda x: x[1], reverse=True)  # Sort by score
        genre_top_tracks = []

        i = 0
        for track in sorted_tracks:
            if track[0]["id"] not in songs_added:
                songs_added.append(track[0]["id"])
                genre_top_tracks.append(track)
                i += 1

            if i == top_n:
                break

        if len(genre_top_tracks) == 0:
            print(f"No tracks found for genre:{genre}, despite starting with {len(tracks)} tracks")

            # Assert that for track in tracks, track is in songs_added
            assert all(track[0]["id"] in songs_added for track in tracks)
        else:
            top_songs[genre] = genre_top_tracks

    return top_songs


def order_songs(top_songs_by_genre, artist_genres):
    genre_lists = artist_genres.values()
    pair_counts = defaultdict(int)

    # Generate all sorted_pairs of genres in each list
    for genre_list in genre_lists:
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


    for genre in top_songs_by_genre.keys():
        if genre not in sequence:
            assert False, f"Genre {genre} not in sequence"

    # Add the songs following the sequence
    final_songs = []
    for genre in sequence:
        if genre in top_songs_by_genre.keys():
            final_songs.extend([item[0] for item in top_songs_by_genre[genre]])

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
    print("Making order...")
    final_songs = order_songs(top_songs_by_genre, artist_genres)
    print("Creating playlist...")
    create_or_update_playlist(sp, user_id, "Trip", final_songs)
    print("Playlist updated successfully!")


if __name__ == "__main__":
    main()
