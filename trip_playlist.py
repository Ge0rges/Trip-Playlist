import spotipy
from spotipy.oauth2 import SpotifyOAuth
from collections import defaultdict, Counter
import numpy as np
from spotipy.client import SpotifyException
import math

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
    Retrieve the user's top tracks.

    :param time_range: The time range for the top tracks. Possible values: 'short_term', 'medium_term', 'long_term'.
    :param limit: The maximum number of tracks to retrieve.
    :return: List of top tracks.
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
        print(f"Processed {i + chunk_size} tracks...")

    return results


def group_by_genre(tracks, play_count, popularity_dict, audio_features, top_tracks):
    genre_dict = defaultdict(list)
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

    # Iterate over tracks and group by genre
    for item in tracks:
        track = item["track"]
        artist_id = track["artists"][0]["id"]
        genres = artist_genres.get(artist_id, [])
        for genre in genres:
            score = play_count[track["id"]] * 2 + popularity_dict[track["id"]]
            if track["id"] in top_tracks:
                score = np.inf
            genre_dict[genre].append((track, score, audio_features[track["id"]]))

    return genre_dict


def calculate_genre_ambiance(genre_tracks):
    valence = 0
    energy = 0
    for track in genre_tracks:
        if track[2] is None:
            return 0.5, 0.5

        energy += track[2]["energy"]
        valence += track[2]["valence"]

    return valence / len(genre_tracks), energy / len(genre_tracks)


def get_top_songs_by_genre(genre_dict, top_n=3):
    top_songs = []
    genre_ambiance = {}

    for genre, tracks in genre_dict.items():
        sorted_tracks = sorted(tracks, key=lambda x: x[1], reverse=True)
        top_tracks = []

        i = 0
        for track in sorted_tracks:
            if track[0] not in top_songs:
                top_songs.append(track[0])
                top_tracks.append(track)
                i += 1

            if i == top_n:
                break

        if len(top_tracks) > 0:
            genre_ambiance[genre] = calculate_genre_ambiance(top_tracks)

    # Sort genres by ambiance (valence, then energy)
    sorted_genres = sorted(genre_ambiance.items(), key=lambda x: (x[1][0], x[1][1]))
    sorted_top_songs = [track[0] for genre in sorted_genres for track in genre_dict[genre[0]][:top_n]]

    return sorted_top_songs


def smooth_transitions(tracks):
    # Fetch audio features for all tracks at once
    track_ids = [track["id"] for track in tracks]
    audio_features_list = sp.audio_features(track_ids)

    # Create list of tuples with track and its corresponding audio features
    tracks_with_features = [
        (track, features) for track, features in zip(tracks, audio_features_list) if features
    ]

    # Step 1: Group the tracks into groups of 3 (assuming len(tracks) % 3 == 0)
    grouped_tracks = [tracks_with_features[i:i + 3] for i in range(0, len(tracks_with_features), 3)]

    # Step 2: Sort within each group of 3 by tempo and key
    sorted_within_groups = [
        sorted(group, key=lambda x: (x[1].get("tempo", 0), x[1].get("key", 0))) for group in grouped_tracks
    ]

    # Step 3: Sort groups based on the average tempo and key of the group
    sorted_groups = sorted(
        sorted_within_groups,
        key=lambda group: (
            sum(track[1].get("tempo", 0) for track in group) / len(group),
            sum(track[1].get("key", 0) for track in group) / len(group)
        )
    )

    # Flatten the sorted list of groups back into a single list of tracks
    final_sorted_tracks = [track[0] for group in sorted_groups for track in group]

    return final_sorted_tracks


def create_or_update_playlist(sp, user_id, playlist_name, tracks):
    playlists = sp.user_playlists(user_id)
    playlist_id = None

    for playlist in playlists["items"]:
        if playlist["name"] == playlist_name:
            playlist_id = playlist["id"]
            break

    if playlist_id:
        sp.playlist_replace_items(playlist_id, [])
    else:
        playlist_id = sp.user_playlist_create(user_id, playlist_name)["id"]

    track_uris = [track["uri"] for track in tracks]
    num_batches = math.ceil(len(track_uris) / 50)
    for i in range(num_batches):
        batch = track_uris[i * 50 : (i + 1) * 50]
        sp.playlist_add_items(playlist_id, batch)


def main():
    print("Getting user ID...")
    user_id = sp.current_user()["id"]
    print("Getting library...")
    library_tracks = get_user_library()
    print("Getting recently played...")
    recently_played = get_recently_played()
    print("Getting counts...")
    play_count = count_plays(recently_played)
    print("Getting popularity...")
    popularity_dict = get_track_popularity([item["track"] for item in library_tracks])
    print("Getting top songs...")
    top_tracks = get_user_top_tracks()
    print("Getting audio features...")
    audio_features_dict = get_audio_features([item["track"] for item in library_tracks])
    print("Grouping track features...")
    genre_dict = group_by_genre(library_tracks, play_count, popularity_dict, audio_features_dict, top_tracks)
    print("Getting top songs by genre...")
    top_songs = get_top_songs_by_genre(genre_dict)
    print("Getting smooth transitions...")
    smoothed_top_songs = smooth_transitions(top_songs)
    print("Creating playlist...")
    create_or_update_playlist(sp, user_id, "Trip", smoothed_top_songs)
    print("Playlist updated successfully!")


if __name__ == "__main__":
    main()
