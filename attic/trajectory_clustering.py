#!/usr/bin/python
import numpy as np
import pylab as pl
import scipy as sp
import random

from behavior_data import data
import behavior_helper as bh

def t_zero( trajectory ):
    trajectory *= 0.0

def t_cummulate( weight, t1, t2 ):
    t1 += weight * t2
        
def t_gaussian( mean, covariance, value ):
    inv = np.linalg.inv( covariance )
    diff = mean - value
    dist = -0.5 * np.dot( np.dot( diff, inv ), diff.transpose() )
    exp = np.exp( np.diagonal( dist ) )
    return np.multiply.reduce( exp )

def expectation( data, means, covariance, f ):
    '''Expectation step'''
    N = len( data )  # Number of trajectories
    M = len( means ) # Number of clusters
    e = np.zeros( ( N, M ) ) # Expectation array
    for n in xrange( N ):
        for m in xrange( M ):
            e[n, m] = f( means[m], covariance, data[n] )
    return e

def maximization( data, e, means, zero, cummulate ):
    '''Maximization step'''
    N = len( data )
    M = len( means )
    for m in xrange( M ):
        zero( means[m] )
        for n in xrange( N ):
            cummulate( e[n, m], means[m], data[n] )
        means[m] /= np.sum( e[:, m] ) # Normalize

def worst_cluster( e ):
    M = e.shape[1]
    with_e = np.sum( np.max( e, 1 ) ) # Sum of best cluster scores
    cluster_scores = np.zeros( (M, ) )
    for m in xrange( M ):
        without_e = e.copy()
        without_e[:, m] = 0. # Exclude current cluster
        cluster_scores[m] = with_e - np.sum( np.max( without_e, 1 ) )
    index = np.argmin( cluster_scores )
    score = np.min( cluster_scores )
    return index, score

def worst_trajectory( e, c_index, c_score, visited, data, covariance ):
    N = data.shape[0]
    M = e.shape[1]

    
    # Sort trajectories by contribution
    traj_contribs = np.sum( e, 1 )
    traj_contribs[visited] = 1E6 # Ignore visited
    sorted_trajs = np.argsort( traj_contribs )

    tmp_e = e.copy()
    tmp_e[:, c_index] = 0.0
    without_e = np.sum( np.max( e, 1 ) ) 
    for k in sorted_trajs:
        for n in xrange( N ):
            tmp_e[n, c_index] = t_gaussian( data[k], covariance, data[n] )
        score = np.sum( np.max( tmp_e, 1 ) ) - without_e
        if score > c_score:
            return k, score
    return -1, 0.0

data = data[:500]
T = data.shape[1]
M = 10
N = data.shape[0]

means = np.zeros( (M, T, 2) )
visited = random.sample( xrange( N ), M )
for m, n in zip( xrange( M ), visited ):
    means[m] = data[n]

covariance = np.array( [[16.0, 0.0], [0.0, 16.0]] )


iterations = 20
for i in xrange( iterations ):
    for j in xrange( 5 ):
        e = expectation( data, means, covariance, t_gaussian )
        maximization( data, e, means, t_zero, t_cummulate )
    if i < iterations - 1:
        c_index, c_score = worst_cluster( e )
        t_index, t_score = worst_trajectory( e, c_index, c_score, visited, data, covariance )
        if t_index == -1:
            break
        means[c_index] = data[t_index]
        visited.append( t_index )

cluster_contribs = np.sum( e, 0 )

print "clusters = np.array( ["
for t in means:
  print "  ["
  for p in t:
    print "    [ %f, %f ]," % ( p[0], p[1] )
  print "  ],"
print "] )"
