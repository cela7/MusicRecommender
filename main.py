import numpy as np
import os
import pandas as pd
import sys
import spotipy
import spotipy.util as util
import threadpoolctl
import torch
import math
import pickle
from spotipy.oauth2 import SpotifyClientCredentials

from cf import CFModels
from cnn import GCNN
from xgb import XGBoost

redirectURI = "https://localhost:8888/callback"
trackInfo = pd.read_hdf( "dataframes/trackInfo.hdf" )
threadpoolctl.threadpool_limits(1, "blas")

def getCredentials():
    f = open( "credentials.txt", "r" )
    clientID = f.readline().strip()
    clientSecret = f.readline().strip()
    f.close()
    return ( clientID, clientSecret )

def NDCG( G, R ):
    DCG = 1 if R[ 0 ] in G else 0
    IDCG = 1
    for i in range( 1, len( R ) ):
        DCG += 1.0 / math.log2( i + 1 ) if R[ i ] in G else 0
    for i in range( 1, len( set( G ) & set( R ) ) ):
        IDCG += 1.0 / math.log( i + 1 )
    return DCG / IDCG

def clicks( G, R ):
    mini = 510
    for i in range( len( R ) ):
        if R[ i ] in G:
            mini = i
            break
    return math.floor( float( mini ) / 10 )

def getBlend( wrmfIds, wrmfScores, cnnIds, cnnScores, itemitemIds, itemitemScores, useruserIds, useruserScores, weights ):
    blendIds = np.array( wrmfIds )
    blendScores = np.array( wrmfScores ) * weights[ 0 ]
    idList = [ cnnIds, itemitemIds, useruserIds ]
    scoreList = [ cnnScores, itemitemScores, useruserScores ]
    for i in range( len( idList ) ):
        for ( id, score ) in zip( idList[ i ], scoreList[ i ] ):
            if id in blendIds:
                blendScores[ np.where( blendIds == id )[ 0 ][ 0 ] ] += score * weights[ i + 1 ]
            else:
                blendIds = np.append( blendIds, id )
                blendScores = np.append( blendScores, score * weights[ i + 1 ] )
    blendIds = blendIds.tolist()
    blendScores = blendScores.tolist()
    blendIds = [ x for _, x in sorted( zip( blendScores, blendIds ), reverse=True ) ]
    blendScores = sorted( blendScores, reverse=True )
    return ( blendIds, blendScores )

def getExtraFeatures( sp, playlist, trackInfo, songFeatures ):
    trackUris = trackInfo.iloc[ playlist ].track_uri.values
    needToRequest = []
    ret = {}
    for uri in trackUris:
        if uri in songFeatures:
            ret[ uri ] = songFeatures[ uri ]
        else:
            needToRequest.append( uri )
    if len( needToRequest ) > 0:
        features = []
        for i in range( 0, len( needToRequest ), 100 ):
            features.extend( sp.audio_features( needToRequest[ i:( i+100 ) ] ) )
        for i in range( len( features ) ):
            if features[ i ] is None:
                # this can happen if the song currently doesn't exist on spotify
                features[ i ] = { "danceability": 0, "energy": 0, "key": 0, "loudness": 0, "mode": 0, "speechiness": 0, "acousticness": 0, "instrumentalness": 0, "liveness": 0, "valence": 0, "tempo": 0, "time_signature": 4 }
            ret[ needToRequest[ i ] ] = features[ i ]
            songFeatures[ needToRequest[ i ] ] = features[ i ]
            with open( "songFeatures.pkl", "wb" ) as f:
                print( "writing current state to file..." )
                pickle.dump( songFeatures, f )
    return ret

def metrics( cfModels, cnn, xgBoost, playlistInfo, trackInfo, songFeatures, sp ):
    test = pd.read_hdf( "dataframes/test.hdf" )
    testLen = test.pid.max()
    maxId = cfModels.wrmfModel.item_factors.shape[ 0 ]
    print( cfModels.wrmfModel.item_factors.shape[ 0 ] )
    minLen = 10
    Dlen = 5
    print( test )
    rPrecs = [ [], [], [], [], [], [] ]
    NDCGS = [ [], [], [], [], [], [] ]
    numClicks = [ [], [], [], [], [], [] ]
    models = [ "WRMF", "CNN", "ItemItem", "UserUser", "Blend", "XGB" ]
    count = 1000
    for i in range( testLen ):
        playlist = test.track_id[ test[ "pid" ] == i ].tolist()
        if len( playlist ) < minLen or max( playlist ) >= maxId or i > len( playlistInfo.index ):
            continue
        pInfo = playlistInfo.iloc[ [ i ] ]
        D = playlist[ :Dlen ]
        G = playlist[ Dlen: ]
        print( f"i, length={len(G)}" )
        wrmfIds, wrmfScores = cfModels.getRecommendations( D, 100 )
        wrmfIdList = wrmfIds.tolist()
        cnnIds, cnnScores = getCNNScores( cnn, D, wrmfIds )
        itemitemIds, itemitemScores = cfModels.itemitem( wrmfIds )
        useruserIds, useruserScores = cfModels.useruser( wrmfIds )
        itemitemIds = [ x for _, x in sorted( zip( itemitemScores, itemitemIds ), reverse=True ) ]
        useruserIds = [ x for _, x in sorted( zip( useruserScores, useruserIds ), reverse=True ) ]
        itemitemScores = sorted( itemitemScores, reverse=True )
        useruserScores = sorted( useruserScores, reverse=True )
        blendIds, blendScores = getBlend( wrmfIds, wrmfScores, cnnIds, cnnScores, itemitemIds, itemitemScores, useruserIds, useruserScores, [ 0.4, 0.1, 0.3, 0.3 ] )
        wrmfList = []
        for p in blendIds:
            if p in wrmfIdList:
                wrmfList.append( wrmfScores[ wrmfIdList.index( p ) ] )
            else:
                wrmfList.append( 0.0 )
        extraFeatures = getExtraFeatures( sp, blendIds, trackInfo, songFeatures )
        xgbIds, xgbScores = xgBoost.getRecommendations( blendIds, wrmfList, playlist, pInfo.iloc[ 0 ], extraFeatures )
        idList = [ wrmfIds, cnnIds, itemitemIds, useruserIds, blendIds, xgbIds ]
        print( "MODEL\t\trPrec\t\tNDCG\t\tClicks" )
        for i in range( len( models ) ):
            rPrecs[ i ].append( float( len( set( G ) & set( idList[ i ][ :len( G ) ] ) ) ) / len( G ) )
            NDCGS[ i ].append( NDCG( G, idList[ i ] ) )
            numClicks[ i ].append( clicks( G, idList[ i ] ) )
            print( f"{models[ i ]}:\t\t{rPrecs[ i ][ len( rPrecs[ i ] ) - 1]}\t{NDCGS[ i ][ len( NDCGS[ i ] ) - 1]}\t{numClicks[ i ][ len( numClicks[ i ] ) - 1]}" )
        count -= 1
        if count == 0:
            break

    for i in range( len( models ) ):
        print( f"Dlen: {Dlen}" )
        print( f"___{models[ i ]}___" )
        print( f"Average RPREC: {np.average( np.array( rPrecs[ i ] ) )}" )
        print( f"Average NDCG: {np.average( np.array( NDCGS[ i ] ) )}" )
        print( f"Average clicks: {np.average( np.array( numClicks[ i ] ) )}" )

def getSongs( sp, playlistId ):
    songs = []
    playlistKeys = [ "name", "collaborative", "pid", "modified_at", "num_tracks", "num_albums", "num_followers", "num_edits", "duration_ms", "num_artists" ]
    ret = sp.playlist( playlistId )
    albums = set()
    artists = set()
    duration = 0
    for track in ret[ "tracks" ][ "items" ]:
        t = track[ "track" ]
        trackUri = "spotify:track:" + t[ "id" ]
        tid = trackInfo.index[ trackInfo[ "track_uri" ] == trackUri ].tolist()
        if len( tid ) > 0:
            songs.append( tid[ 0 ] )
            albums.add( t[ "album" ][ "name" ] )
            artists.add( t[ "artists" ][ 0 ][ "name" ] )
            duration += int( t[ "duration_ms" ] )
        else:
            print( f"skipped {t[ 'name' ]}, couldn't find track id" )
        print( f"name: {t[ 'name' ]}, uri: {trackUri}, tid: {tid}" )
    pInfo = [ ret[ "name" ], ret[ "collaborative" ], 0, "", len( songs ), len( albums ), 0, 0, duration, len( artists ) ]
    pInfo = pd.DataFrame( [ pInfo ], columns=playlistKeys )
    return ( songs, pInfo )

def displayResults( ids, scores ):
    song_names = []
    artists = []
    for track_id in ids:
        song_names.append( trackInfo.track_name[ trackInfo[ "track_id" ] == track_id ].tolist()[ 0 ] )
        artists.append( trackInfo.artist_name[ trackInfo[ "track_id" ] == track_id ].tolist()[ 0 ] )
    results = pd.DataFrame( { "name": song_names, "artist": artists, "score": scores } )
    results = results.sort_values( "score", ascending=False )
    print( results )

def getCNNScores( model, playlist, wrmfIds ):
    repetitions = 10
    totalIds = []
    if len( playlist ) < 10:
        playlist.extend( wrmfIds[ :( 10 - len( playlist ) ) ] )
    pl = torch.Tensor( [ playlist[ :10 ] ] ).to( torch.long )
    for i in range( repetitions ):
        if torch.cuda.is_available():
            pl = pl.cuda()
        output = model( pl )
        ids = torch.argmax( output, dim=1 ).tolist()[ 0 ]
        newIds = []
        for i in range( len( ids ) ):
            newId = ids[ i ]
            while newId in newIds or newId in totalIds:
                newId += 1
            newIds.append( newId )
        totalIds.extend( newIds )
        pl = torch.Tensor( [ newIds ] ).to( torch.long )
    scores = np.arange( 1, 0, -1.0 / ( repetitions * 10 ) )
    return ( totalIds, scores )

def main():
    playlists = pd.read_hdf( "dataframes/playlists.hdf" )
    playlistInfo = pd.read_hdf( "dataframes/playlistInfo.hdf" )
    trackInfo = pd.read_hdf( "dataframes/trackInfo.hdf" )
    with open( "songFeatures.pkl", "rb" ) as f:
        songFeatures = pickle.load( f )
    numTracks = trackInfo.track_id.max() + 1
    cfModels = CFModels( playlists, playlistInfo, trackInfo )
    cfModels.loadWRMF()
    xgBoost = XGBoost( playlists, playlistInfo, trackInfo )
    xgBoost.load()
    kernelSize = 30
    embeddingSize = 200
    with open( "embeddings.npy", "rb" ) as f:
        embeddings = np.load( f )
    model = GCNN( numTracks, embeddingSize, embeddings, kernelSize, 50, 6, 5, 0.1 )
    model.load_state_dict( torch.load( "model_checkpoint_itr_24" )[ "model_state_dict" ] )
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    if len( sys.argv ) > 1 and sys.argv[ 1 ] == "metrics":
        print( "Getting metrics..." )
        clientId, clientSecret = getCredentials()
        clientCredentials = SpotifyClientCredentials( client_id=clientId, client_secret=clientSecret )
        sp = spotipy.Spotify( client_credentials_manager=clientCredentials )
        metrics( cfModels, model, xgBoost, playlistInfo, trackInfo, songFeatures, sp )
        exit()
    verbose = 0
    if len( sys.argv ) > 1 and sys.argv[ 1 ] == "verbose":
        verbose = 1
    clientId, clientSecret = getCredentials()
    scope = 'playlist-read-private playlist-read-collaborative'
    username = input( "username: " ).strip()
    token = util.prompt_for_user_token( username, scope, client_id=clientId, client_secret=clientSecret, redirect_uri=redirectURI )
    if token:
        sp = spotipy.Spotify( auth=token )
        playlistId = input( "Enter id of playlist to get recommendations: " )
        ( songs, pInfo ) = getSongs( sp, playlistId )
        if verbose:
            print( pInfo )
            print( songs )
        wrmfIds, wrmfScores = cfModels.getRecommendations( songs, 100 )
        if verbose:
            print( "WRMF:" )
            displayResults( wrmfIds, wrmfScores )
            print( "CNN:" )
        else:
            print( "WRMF finished..." )
        cnnIds, cnnScores = getCNNScores( model, songs, wrmfIds )
        if verbose:
            displayResults( cnnIds, cnnScores )
            print( "item-item" )
        else:
            print( "CNN finished..." )
        itemIds, itemScores = cfModels.itemitem( wrmfIds )
        if verbose:
            displayResults( itemIds, itemScores )
            print( "user-user" )
        else:
            print( "item-item finished..." )
        userIds, userScores = cfModels.useruser( wrmfIds )
        if verbose:
            displayResults( userIds, userScores )
        else:
            print( "user-user finished..." )
        blendIds, blendScores = getBlend( wrmfIds, wrmfScores, cnnIds, cnnScores, itemIds, itemScores, userIds, userScores, [ 0.4, 0.1, 0.3, 0.3 ] )
        if verbose:
            displayResults( blendIds, blendScores )
        else:
            print( "blend finished..." )
        wrmfList = []
        wrmfIds = wrmfIds.tolist()
        for p in blendIds:
            if p in wrmfIds:
                wrmfList.append( wrmfScores[ wrmfIds.index( p ) ] )
            else:
                wrmfList.append( 0.0 )
        extraFeatures = getExtraFeatures( sp, blendIds, trackInfo, songFeatures )
        xgbIds, xgbScores = xgBoost.getRecommendations( blendIds, wrmfList, songs, pInfo.iloc[ 0 ], extraFeatures )
        pd.set_option( "display.max_rows", None )
        print( "Recommendations: " )
        displayResults( xgbIds, xgbScores )
        
    else:
        print( f"Can't get token for {username}" )
    

if __name__ == "__main__":
    main()