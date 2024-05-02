import numpy as np
import pandas as pd
import torch
import time
from xgboost import XGBRanker

from cf import CFModels
from cnn import GCNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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

class XGBoost():
    def __init__( self, playlists, playlistInfo, trackInfo ):
        self.model = XGBRanker( n_estimators=150, max_depth=10, tree_method="hist", device="cpu", booster="gbtree", objective="rank:pairwise", learning_rate=0.01 )
        self.albumEncoder = LabelEncoder()
        self.albumEncoder.fit( trackInfo.album_name.values )
        self.trackInfo = trackInfo
        self.playlistInfo = playlistInfo
        self.playlists = playlists
    
    def addPlaylistSongFeatures( self, playlist, df ):
        albumNames = df.loc[ :, "album_name" ].tolist()
        plAlbumNames = playlist.loc[ :, "album_name" ].tolist()
        plIds = playlist.loc[ :, "track_id" ]
        dfIds = df.loc[ :, "track_id" ]
        plTracks = self.trackInfo.iloc[ plIds ]
        dfTracks = self.trackInfo.iloc[ dfIds ]
        plArtists = plTracks.artist_name.values.tolist()
        dfArtists = dfTracks.artist_name.values.tolist()
        albumOccurrence = []
        artistOccurrence = []
        for i in range( len( albumNames ) ):
            albumOccurrence.append( plAlbumNames.count( albumNames[ i ] ) / len( plAlbumNames ) )
            artistOccurrence.append( plArtists.count( dfArtists[ i ] ) / len( plArtists ) )
        df[ "album_occurrence" ] = albumOccurrence
        df[ "artist_occurrence" ] = artistOccurrence
        return df
    
    def prepareData( self, stage2, cfModels, cnn ):
        stage2[ "album_name" ] = self.albumEncoder.transform( stage2.album_name.values )
        stage2 = stage2.drop( columns=[ "playlist_name", "track_name" ] )

        maxId = cfModels.wrmfModel.item_factors.shape[ 0 ]
        pids = sorted( set( stage2.pid.tolist() ) )
        train = []
        label = []
        qid = []
        curQid = 0
        minLen = 1
        print( len( pids ) )
        for pid in pids:
            if curQid % 100 == 0:
                print( f"cur qid: {curQid} / ~14000" )
            playlist = stage2[ stage2[ "pid" ] == pid ].copy()
            if len( playlist ) < 30:
                # skip playlists that don't have a sufficient amount of songs for both
                # selecting relevent songs and generating 
                continue
            wrmfIds, wrmfScores = cfModels.getRecommendations( playlist.track_id.values, 200, False )
            wrmfIds = wrmfIds.tolist()
            # do 20-20 sampling, where 20 songs from the playlist are sampled
            # and labeled as relevent (1), and 20 songs are picked randomly
            # and deemed not relevent (0)
            relevant = playlist.sample( n=min( len( playlist ), 20 ) )
            relevant = relevant[ relevant[ "track_id" ] < maxId ]
            relevant.drop( columns=[ "pid" ], inplace=True )
            relevant = self.addPlaylistSongFeatures( playlist, relevant )
            rest = playlist.copy()
            restCond = rest[ "track_id" ].isin( relevant[ "track_id" ] )
            rest.drop( rest[ restCond ].index, inplace=True )
            irrelevant = stage2.sample( n=20 )
            irrelevant = irrelevant[ irrelevant[ "track_id" ] < maxId ]
            cond = irrelevant[ "track_id" ].isin( playlist[ "track_id" ] )
            irrelevant.drop( irrelevant[ cond ].index, inplace=True )
            irrelevant.drop( columns=[ "pid" ], inplace=True )
            irrelevant = self.addPlaylistSongFeatures( playlist, irrelevant )
            wrmfList = []
            for p in relevant.track_id.values:
                if p in wrmfIds:
                    wrmfList.append( wrmfScores[ wrmfIds.index( p ) ] )
                else:
                    wrmfList.append( 0.0 )
            relevant[ "wrmf" ] = wrmfList
            wrmfList = []
            for p in irrelevant.track_id.values:
                if p in wrmfIds:
                    wrmfList.append( wrmfScores[ wrmfIds.index( p ) ] )
                else:
                    wrmfList.append( 0.0 )
            irrelevant[ "wrmf" ] = wrmfList
            train.extend( relevant.to_numpy().tolist() )
            train.extend( irrelevant.to_numpy().tolist() )
            label.extend( np.ones( len( relevant ) ).tolist() )
            label.extend( np.zeros( len( irrelevant ) ).tolist() )
            qid.extend( np.full( len( relevant ) + len( irrelevant ), curQid ).tolist() )
            curQid += 1
        self.train( np.array( train, dtype=object ), np.array( label, dtype=bool ), np.array( qid, dtype=int ) )

    def train( self, train, label, qid ):
        self.model.fit( train, label, qid=qid, verbose=True )
        self.model.save_model( "xgbModel.json" )
    
    def load( self ):
        self.model.load_model( "xgbModel.json" )
    
    def getRecommendations( self, ids, wrmfList, playlist, pInfo, extraFeatures ):
        xgbIds = ids
        xgbScores = []
        data = []
        pos = 0
        tracks = self.trackInfo.iloc[ ids ]
        pl = self.trackInfo.iloc[ playlist ]
        plAlbums = pl.album_name.values.tolist()
        plArtists = pl.artist_name.values.tolist()
        albumNames = self.albumEncoder.transform( tracks.album_name.values )
        for song in ids:
            #print( song )
            track = self.trackInfo.iloc[ [ song ] ].iloc[ 0 ]
            ex = extraFeatures[ track[ "track_uri" ] ]
            album_occurrence = plAlbums.count( tracks.album_name.values[ pos ] ) / len( plAlbums )
            artist_occurrence = plArtists.count( tracks.artist_name.values[ pos ] ) / len( plArtists )
            # stage2Columns = [ "pid", "playlist_name", "playlist_duration", "track_id", "pos", "track_name", "album_name", "duration", "danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "time_signature" ]
            row = np.array( [ pInfo[ "duration_ms" ], song, pos, albumNames[ pos ], track[ "duration_ms" ], ex[ "danceability" ], ex[ "energy" ], ex[ "key" ], ex[ "loudness" ], ex[ "mode" ], ex[ "speechiness" ], ex[ "acousticness" ], ex[ "instrumentalness" ], ex[ "liveness" ], ex[ "valence" ], ex[ "tempo" ], ex[ "time_signature" ], album_occurrence, artist_occurrence, wrmfList[ pos ] ] )
            data.append( row )
            pos += 1
        xgbScores = self.model.predict( np.array( data ) )
        xgbIds = [ x for _, x in sorted( zip( xgbScores, xgbIds ), reverse=True ) ]
        xgbScores = sorted( xgbScores, reverse=True )
        return ( xgbIds, xgbScores )

if __name__ == "__main__":
    stage2 = pd.read_hdf( "dataframes/stage2.hdf" )
    playlists = pd.read_hdf( "dataframes/playlists.hdf" )
    trackInfo = pd.read_hdf( "dataframes/trackInfo.hdf" )
    playlistInfo = pd.read_hdf( "dataframes/playlistInfo.hdf" )
    cfModels = CFModels( playlists, playlistInfo, trackInfo )
    cfModels.loadWRMF()
    kernelSize = 30
    embeddingSize = 200
    numTracks = trackInfo.track_id.max() + 1
    with open( "embeddings.npy", "rb" ) as f:
        embeddings = np.load( f )
    model = GCNN( numTracks, embeddingSize, embeddings, kernelSize, 50, 6, 5, 0.1 )
    model.load_state_dict( torch.load( "model_checkpoint_itr_24" )[ "model_state_dict" ] )
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    xgboost = XGBoost( playlists, playlistInfo, trackInfo )
    print( stage2 )
    xgboost.prepareData( stage2, cfModels, model )

