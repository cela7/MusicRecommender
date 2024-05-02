from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import CosineRecommender
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import scipy.sparse as sp
import numpy as np
import threadpoolctl
import time

class CFModels():
    # Collaborative filtering models (WRMF, item-item, user-user)
    def __init__( self, playlists, playlistInfo, trackInfo ):
        self.numPlaylists = playlistInfo.pid.max() + 1
        self.numTracks = trackInfo.track_id.max() + 1
        self.R = sp.coo_matrix( ( np.ones( len( playlists ) ), ( playlists.pid, playlists.track_id ) ), shape=( self.numPlaylists, self.numTracks ) )
        self.R = self.R.tocsr()
        self.wrmfModel = AlternatingLeastSquares( factors=200, regularization=0.001, alpha=100, calculate_training_loss=True )
        self.maxPop = 4457
        self.storedCols = {}
    
    def getMaxPop( self ):
        # Only run this when getting the maximum popularity for the first time, otherwise use the value received from this
        # as a constant, this method takes a long time to run
        print( "getting max pop..." )
        for j in range( self.numTracks ):
            rows = self.R.indptr.searchsorted( *(self.R.indices==j).nonzero(), "right" ) - 1
            if ( len( rows ) > self.maxPop ):
                self.maxPop = len( rows )
        print( self.maxPop )
        return self.maxPop

    def trainWRMF( self ):
        self.wrmfModel.fit( self.R )
        with open( "wrmfModel.npy", "wb" ) as f:
            self.wrmfModel.save( f )
        with open( "embeddings.npy", "wb" ) as f:
            np.save( f, self.wrmfModel.item_factors )
    def loadWRMF( self ):
        with open( "wrmfModel.npy", "rb" ) as f:
            self.wrmfModel = self.wrmfModel.load( f )
    
    def useruser( self, playlist, beta=0.9 ):
        scores = []
        pl = sp.csr_matrix( ( np.ones( len( playlist ) ), playlist, [ 0, len( playlist ) ] ), shape=( 1, self.numTracks ) )
        simTable = {}
        for j in playlist:
            # Select rows that have a 1 at column j
            rows = self.R.indptr.searchsorted( *(self.R.indices==j).nonzero(), "right" ) - 1
            popularity = len( rows ) / self.maxPop
            score = 0
            for i in rows:
                if i in simTable:
                    score += simTable[ i ]
                else:
                    similarity = cosine_similarity( pl, self.R[ i, : ] )[ 0 ][ 0 ]
                    score += similarity
                    simTable[ i ] = similarity
            scores.append( score )
        scores = np.array( scores ) * ( popularity ** ( -( 1 - beta ) ) )
        scores /= max( scores )
        return playlist, scores

    def itemitem( self, playlist, beta=0.6 ):
        scores = []
        for j in playlist:
            score = 0
            rows = self.R.indptr.searchsorted( *(self.R.indices==j).nonzero(), "right" ) - 1
            popularity = len( rows ) / self.maxPop
            if j in self.storedCols:
                RjT = self.storedCols[ j ]
            else:
                RjT = self.R[ :, j ].T
                self.storedCols[ j ] = RjT
            for jp in playlist:
                if jp == j:
                    continue
                if jp in self.storedCols:
                    score += cosine_similarity( RjT, self.storedCols[ jp ] )[ 0 ][ 0 ]
                else:
                    RjpT = self.R[ :, jp ].T
                    self.storedCols[ jp ] = RjpT
                    score += cosine_similarity( RjT, RjpT )[ 0 ][ 0 ]
            scores.append( score )
        scores = np.array( scores ) * ( popularity ** ( -( 1 - beta ) ) )
        scores /= max( scores )
        return playlist, scores
    
    def getRecommendations( self, playlist, N, filterAlreadyLiked=True ):
        onehot = sp.csr_matrix( ( np.ones( len( playlist ) ), playlist, [ 0, len( playlist ) ] ), shape=( 1, self.numTracks ) )
        ids, scores = self.wrmfModel.recommend( 0, onehot, N=N, recalculate_user=True, filter_already_liked_items=filterAlreadyLiked )
        return ( ids, scores )

if __name__ == "__main__":
    threadpoolctl.threadpool_limits(1, "blas")

    playlists = pd.read_hdf( "dataframes/playlists.hdf" )
    playlistInfo = pd.read_hdf( "dataframes/playlistInfo.hdf" )
    trackInfo = pd.read_hdf( "dataframes/trackInfo.hdf" )

    cfModels = CFModels( playlists, playlistInfo, trackInfo )
    cfModels.trainWRMF()
    #cfModels.loadWRMF()
    #cfModels.wrmfModel.item_factors = embeddings
    #print( cfModels.wrmfModel.item_factors.shape )
    pl = [336244, 381671, 352630, 58105, 336254, 283166, 166652, 336252, 170944]
    print( "getting recommendations..." )
    ids, scores = cfModels.getRecommendations( pl, 10 )
    song_names = []
    artists = []
    for track_id in ids:
        song_names.append( trackInfo.track_name[ trackInfo[ "track_id" ] == track_id ].tolist()[ 0 ] )
        artists.append( trackInfo.artist_name[ trackInfo[ "track_id" ] == track_id ].tolist()[ 0 ] )
    results = pd.DataFrame( { "name": song_names, "artist": artists, "score": scores } )
    results = results.sort_values( "score", ascending=False )
    print( results )
    print( "user-user scores: " )
    recIds = ids
    ids, scores = cfModels.useruser( recIds )
    song_names = []
    artists = []
    for track_id in ids:
        song_names.append( trackInfo.track_name[ trackInfo[ "track_id" ] == track_id ].tolist()[ 0 ] )
        artists.append( trackInfo.artist_name[ trackInfo[ "track_id" ] == track_id ].tolist()[ 0 ] )
    results = pd.DataFrame( { "name": song_names, "artist": artists, "score": scores } )
    results = results.sort_values( "score", ascending=False )
    print( results )
    print( "item-item scores: " )
    ids, scores = cfModels.itemitem( recIds )
    song_names = []
    artists = []
    for track_id in ids:
        song_names.append( trackInfo.track_name[ trackInfo[ "track_id" ] == track_id ].tolist()[ 0 ] )
        artists.append( trackInfo.artist_name[ trackInfo[ "track_id" ] == track_id ].tolist()[ 0 ] )
    results = pd.DataFrame( { "name": song_names, "artist": artists, "score": scores } )
    results = results.sort_values( "score", ascending=False )
    print( results )