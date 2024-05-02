import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
import pandas as pd
import scipy.sparse as sp
import numpy as np
import os
import threadpoolctl
import random

class PlaylistDataset( Dataset ):
    def __init__( self, data, targets ):
        self.data = data
        self.targets = targets
    def __len__( self ):
        return len( self.data )
    def __getitem__( self, idx ):
        return ( torch.Tensor( self.data[ idx ] ).to( torch.long ), torch.Tensor( self.targets[ idx ] ).to( torch.long ) )

def preprocess( playlists, playlistInfo, seq_len ):
    # Create a new dataframe with all playlists split up into chunks of size seq_len
    # All playlists with < seq_len*2 are removed, each data point has a seq_len size
    # array containing a seq_len sized subsection of an existing playlist, and a
    # corresponding target which is the following seq_len sized subsection of that
    # playlist
    data = []
    targets = []
    for idx in range( playlistInfo.pid.max() ):
        playlist = playlists.track_id[ playlists[ "pid" ] == idx ].tolist()
        if idx % 100 == 0:
            print( f"preprocessing... {idx}/{playlistInfo.pid.max()}" )
        if len( playlist ) < seq_len * 2:
            continue
        for k in range( 0, len( playlist ) - seq_len*2, 10 ):
            data.append( playlist[ k:k+seq_len ] )
            targets.append( playlist[ k+seq_len:k+seq_len*2 ] )
    return ( data, targets )

class GCNN( nn.Module ):
    def __init__( self, vocab_size, embedding_size, embeddings, kernels, out_channels, layers, res_block_cnt, dropout=0.1 ):
        super( GCNN, self ).__init__()
        self.res_block_cnt = res_block_cnt
        self.embed = nn.Embedding.from_pretrained( torch.FloatTensor( embeddings ), freeze=False )
        self.leftpad = nn.ConstantPad1d( ( kernels - 1, 0 ), 0 )
        self.conv_0 = nn.Conv1d( in_channels=embedding_size, out_channels=out_channels, kernel_size=kernels )
        self.b_0 = nn.Parameter( torch.zeros( out_channels, 1 ) )
        self.conv_gate_0 = nn.Conv1d( in_channels=embedding_size, out_channels=out_channels, kernel_size=kernels )
        self.c_0 = nn.Parameter( torch.zeros( out_channels, 1 ) )
        self.convs = nn.ModuleList( [ nn.Conv1d( in_channels=out_channels, out_channels=out_channels, kernel_size=kernels ) for _ in range( layers ) ] )
        self.bs = nn.ParameterList( [ nn.Parameter( torch.zeros( out_channels, 1 ) ) for _ in range( layers ) ] )
        self.conv_gates = nn.ModuleList( [ nn.Conv1d( in_channels=out_channels, out_channels=out_channels, kernel_size=kernels ) for _ in range( layers ) ] )
        self.cs = nn.ParameterList( [ nn.Parameter( torch.zeros( out_channels, 1 ) ) for _ in range( layers ) ] )
        self.fc = nn.Linear( out_channels, vocab_size )
        self.dropout = nn.Dropout( dropout )
        self.output = nn.AdaptiveLogSoftmaxWithLoss( embedding_size, vocab_size, cutoffs=[ round( vocab_size/15 ), 3*round( vocab_size/15 ) ], div_value=4 )
    def forward( self, x ):
        x = self.embed( x )
        x.transpose_( 1, 2 )
        x = self.leftpad( x )
        xa = self.conv_0( x ) + self.b_0
        xb = self.conv_gate_0( x ) + self.c_0
        h = xa * F.sigmoid( xb )
        res = h
        for i, ( conv, convGate ) in enumerate( zip( self.convs, self.conv_gates ) ):
            h = self.leftpad( h )
            xa = conv( h ) + self.bs[ i ]
            xb = convGate( h ) + self.cs[ i ]
            h = xa * F.sigmoid( xb )
            if i % self.res_block_cnt == 0:
                h += res
                res = h
        h.transpose_( 1, 2 )
        h = self.fc( h )
        h.transpose_( 1, 2 )
        return h

def validate( model, criterion, val_dataloader ):
    # get validation loss and save checkpoint
    print( "Validating..." )
    model.eval()
    correct = 0
    total = 0
    valLoss = 0
    num = 0
    with torch.no_grad():
        for batch_idx, ( data, target ) in enumerate( val_dataloader ):
            data = data.cuda()
            target = target.cuda()
            output = model( data )
            loss = criterion( output, target )
            pred = torch.argmax( output, dim=1 ).cuda()
            correct += ( pred == target ).sum().item() 
            total += target.size( 0 )
            valLoss += loss.item()
            num += 1
    valAccuracy = ( 100 * correct ) / total
    valLoss /= num
    print( f"Validation accuracy: {valAccuracy}" )
    print( f"Validation loss: {valLoss}" )
    return valLoss

def train( numTracks, embeddingSize, embeddings, kernelSize, train_dataloader, val_dataloader, load="" ):
    iterations = 50
    model = GCNN( numTracks, embeddingSize, embeddings, kernelSize, 50, 6, 5, 0.1 )
    if torch.cuda.is_available():
        model.cuda()
    criterion = nn.CrossEntropyLoss( ignore_index=0 )
    optimizer = optim.SGD( model.parameters(), lr=0.01, momentum=0.99 )
    if load != "":
        checkpoint = torch.load( load )
        model.load_state_dict( checkpoint[ "model_state_dict" ] )
        optimizer.load_state_dict( checkpoint[ "optimizer_state_dict" ] )
        for g in optimizer.param_groups:
            g['lr'] = 0.001
        print( f"last val loss: {checkpoint[ 'valLoss' ]}" )

    st = time.time()
    losses = []
    for i in range( iterations ):
        model.train()
        for batch_idx, ( data, target ) in enumerate( train_dataloader ):
            #print( f"data: {data}, target: {target}" )
            optimizer.zero_grad()
            #data = torch.transpose( data, 0, 1 ).cuda()
            data = data.cuda()
            output = model( data )
            loss = criterion( output, target.cuda() )
            loss.backward()
            torch.nn.utils.clip_grad_value_( model.parameters(), 0.07 )
            optimizer.step()

            if batch_idx % 100 == 0:
                elapsed = time.time() - st
                pred = torch.argmax( output, dim=1 )
                #losses.append( loss.item() )
                print( f"batch_idx: {batch_idx}, iteration: {i}/{iterations}\tloss: {loss.item()}\ttime: {elapsed/60} min, memory: {torch.cuda.memory_allocated()}" )
                print( f"output of playlist: {pred}")
        if i % 3 == 0:
            # get validation loss and save checkpoint
            valLoss = validate( model, criterion, val_dataloader )
            
            checkpoint = {
                "epoch": i,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "valLoss": valLoss,
                "losses": losses
            }
            torch.save( checkpoint, f"model_checkpoint_itr_{i}" )
    print( f"final train loss: {loss.item()}, elapsed time: {(time.time()-st)/60}")
    torch.save( model.state_dict(), "model_state_dict_full.pt" )

if __name__ == "__main__":
    playlists = pd.read_hdf( "dataframes/playlists.hdf" )
    playlistInfo = pd.read_hdf( "dataframes/playlistInfo.hdf" )
    trackInfo = pd.read_hdf( "dataframes/trackInfo.hdf" )
    validation = pd.read_hdf( "dataframes/validation.hdf" )

    #numPlaylists = playlistInfo.pid.max() + 1
    numTracks = trackInfo.track_id.max() + 1

    kernelSize = 30
    embeddingSize = 200
    with open( "embeddings.npy", "rb" ) as f:
        embeddings = np.load( f )
    # print( embeddings.shape )

    data, targets = preprocess( playlists, playlistInfo, 10 )
    #data, targets = ( [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], [[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]] )
    valData, valTargets = preprocess( validation, playlistInfo, 10 )
    train_dataset = PlaylistDataset( data, targets )
    val_dataset = PlaylistDataset( valData, valTargets )
    train_dataloader = DataLoader( train_dataset, batch_size=16, shuffle=True )
    val_dataloader = DataLoader( val_dataset, batch_size=10, shuffle=False )
    train( numTracks, embeddingSize, embeddings, kernelSize, train_dataloader, val_dataloader, "model_checkpoint_itr_6_prev" )