import sys
import json
import time
import os
import numpy as np
import scipy.sparse as sparse
import pandas as pd
import pickle

from spotipy.oauth2 import SpotifyClientCredentials
import spotipy

# Shorten the dataset, improves training and eval speed but reduces accuracy
MAX_COUNT = 202

playlistKeys = [ "name", "collaborative", "pid", "modified_at", "num_tracks", "num_albums", "num_followers", "num_edits", "duration_ms", "num_artists" ]
trackKeys = [ "artist_name", "track_uri", "artist_uri", "track_name", "album_uri", "duration_ms", "album_name" ]

def processExtra( data, trackUriToId, sp, storagePath, dfPath ):
    extraColumns = [ "pid", "playlist_name", "playlist_duration", "track_id", "pos", "track_name", "album_name", "duration", "danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "time_signature" ]
    #stage2Columns = [ "pid", "playlist_name", "playlist_duration", "track_id", "pos", "track_name", "album_name", "duration" ]
    # For stage 2 we need to grab more song information, so we loop through and
    # request spotify for information
    requestsPer30 = 90
    songFeatures = {}
    # loop through once to get features, needed to reduce API calls
    trackList = []
    seen = set()
    last = data[ -1 ]
    requests = 0
    st = time.time()
    # load from backup just in case spotify decided to close connection
    if os.path.isfile( storagePath ):
        f = open( storagePath, "rb" )
        songFeatures = pickle.load( f )
        for key in songFeatures:
            if songFeatures[ key ] is not None:
                seen.add( key )
            else:
                print( f"none: {key}" )
        f.close()
        print( "loaded features from file" )
    # for all songs in the dataset, request spotify for extra features
    for song in data:
        if song[ 3 ] not in seen:
            trackList.append( song[ 3 ] )
        seen.add( song[ 3 ] )
        if len( trackList ) == 100 or song == last:
            dt = time.time() - st
            if requests >= requestsPer30 and dt < 30:
                print( f"Hit max requests per 30s, waiting {dt}s..." )
                time.sleep( dt )
                requests = 0
                st = time.time()
            features = sp.audio_features( trackList )
            for i in range( len( trackList ) ):
                if features[ i ] is not None:
                    songFeatures[ trackList[ i ] ] = features[ i ]
                else:
                    # this can happen if the song currently doesn't exist on spotify
                    songFeatures[ trackList[ i ] ] = { "danceability": 0, "energy": 0, "key": 0, "loudness": 0, "mode": 0, "speechiness": 0, "acousticness": 0, "instrumentalness": 0, "liveness": 0, "valence": 0, "tempo": 0, "time_signature": 4 }
            with open( storagePath, "wb" ) as f:
                print( "writing current state to file..." )
                pickle.dump( songFeatures, f )
            trackList.clear()
            requests += 1
            time.sleep( 5 )
    # loop through again to update stage 2 songs
    features = [ "danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "time_signature" ]
    for song in data:
        for feature in features:
            song.append( songFeatures[ song[ 3 ] ][ feature ] )
     
    df = pd.DataFrame( data, columns=extraColumns )
    df.track_id = df.track_id.map( trackUriToId )
    df.to_hdf( dfPath, key="abc" )

def process_playlists( path, sp ):
    filenames = os.listdir( path )
    count = 0

    # Here we assume that each file contains a randomly sampled group of 1000 playlists.
    # The size of the validation set is 2000, so we choose 2 random files out of 200
    # total files to make up the validation set. Additionally, we choose 20 random files
    # out of 2000 total files to make up the second stage training set
    fileRange = np.random.choice( 200, 22, replace=False )
    validationFiles = fileRange[ :2 ]
    stage2Files = fileRange[ 2: ]
    # use hardcoded random for testing
    validationFiles = [ 177, 25 ]
    testFiles = [ 201, 202 ]
    stage2Files = [ 30, 99, 7, 6, 9, 118, 196, 34, 151, 1, 199, 90, 105, 14, 136, 174, 16, 156, 11, 186 ]

    playlistsData = []
    tracksData = []
    playlists = []
    validation = []
    stage2 = []
    test = []
    tracks = set()
    for filename in sorted( filenames ):
        count += 1
        print( f"=== {count} / {len( filenames )} ===" )
        if filename.startswith( "mpd.slice." ) and filename.endswith( ".json" ):
            fullpath = os.sep.join( ( path, filename ) )
            f = open( fullpath )
            js = f.read()
            f.close()
            mpd_slice = json.loads( js )
            for playlist in mpd_slice[ "playlists" ]:
                #print( f"{playlist[ 'pid' ]}: {playlist[ 'name' ]}" )
                playlistsData.append( [ playlist[ key ] for key in playlistKeys ] )
                for i, track in enumerate( playlist[ "tracks" ] ):
                    #print( f"{i}: {track['track_uri']}, {track['track_name']}" )
                    if count in validationFiles:
                        validation.append( [ playlist[ "pid" ], track[ "track_uri" ], track[ "pos" ] ] )
                    elif count in stage2Files:
                        stage2.append( [ playlist[ "pid" ], playlist[ "name" ], playlist[ "duration_ms" ], track[ "track_uri" ], track[ "pos" ], track[ "track_name" ], track[ "album_name" ], track[ "duration_ms" ] ] )
                    elif count in testFiles:
                        test.append( [ playlist[ "pid" ], playlist[ "name" ], playlist[ "duration_ms" ], track[ "track_uri" ], track[ "pos" ], track[ "track_name" ], track[ "album_name" ], track[ "duration_ms" ] ] )
                    else:
                        playlists.append( [ playlist[ "pid" ], track[ "track_uri" ], track[ "pos" ] ] )
                    if track[ "track_uri" ] not in tracks:
                        tracks.add( track[ "track_uri" ] )
                        tracksData.append( [ track[ key ] for key in trackKeys ] )
        if MAX_COUNT == count:
            break
    
    playlistInfoDf = pd.DataFrame( playlistsData, columns=playlistKeys )
    trackInfoDf = pd.DataFrame( tracksData, columns=trackKeys )
    trackInfoDf[ "track_id" ] = trackInfoDf.index
    trackUriToId = trackInfoDf.set_index( "track_uri" ).track_id
    playlistsDf = pd.DataFrame( playlists, columns=[ "pid", "track_id", "pos" ] )
    playlistsDf.track_id = playlistsDf.track_id.map( trackUriToId )

    validationDf = pd.DataFrame( validation, columns=[ "pid", "track_id", "pos" ] )
    validationDf.track_id = validationDf.track_id.map( trackUriToId )

    # Need to process more information for stage 2
    processExtra( stage2, trackUriToId, sp, "songFeatures.pkl", "dataframes/stage2.hdf" )

    playlistInfoDf.to_hdf( "dataframes/playlistInfo.hdf", key="abc" )
    trackInfoDf.to_hdf( "dataframes/trackInfo.hdf", key="abc" )
    playlistsDf.to_hdf( "dataframes/playlists.hdf", key="abc" )
    validationDf.to_hdf( "dataframes/validation.hdf", key="abc" )
    # Process more info for test as well
    processExtra( test, trackUriToId, sp, "testFeatures.pkl", "dataframes/test.hdf" )

def getCredentials():
    f = open( "credentials.txt", "r" )
    clientID = f.readline().strip()
    clientSecret = f.readline().strip()
    f.close()
    return ( clientID, clientSecret )

if __name__ == "__main__":
    path = sys.argv[1]
    dirs = os.listdir( "." )
    if ( "dataframes" not in dirs ):
        os.makedirs( "dataframes" )
    clientId, clientSecret = getCredentials()
    clientCredentials = SpotifyClientCredentials( client_id=clientId, client_secret=clientSecret )
    sp = spotipy.Spotify( client_credentials_manager=clientCredentials )
    process_playlists( path, sp )