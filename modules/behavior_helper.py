import numpy as np
import pylab as pl
from sklearn.cluster import KMeans

palettes = {}

def generate_palette( count ):
  if count in palettes:
    return palettes[count]
  samples = np.random.randint( 0, 255, ( 1000, 3 ) ) 
  km = KMeans( count, init = "k-means++" )
  km.fit( samples )
  palettes[count] = km.cluster_centers_ / 255.0
  return palettes[count]

def plot_trajectories( data ):
  if len( data ) < 50:
    palette = generate_palette( len( data ) )
    for n, t in enumerate( data ):
      t = np.array( t )
      pl.plot( t[:,0], t[:,1], "-", c = palette[n] )
      pl.plot( t[-1,0], t[-1,1], "o", markersize = 5.0, c = palette[n] )
  else:
    for t in data:
      t = np.array( t )
      pl.plot( t[:,0], t[:,1], "-" )

def plot_time_model( data ):
  palette = generate_palette( len( data ) )
  for n, t in enumerate( data ):
    t = np.array( t )
    pl.plot( t[:,0], t[:,1], "o", c = palette[n] )

def plot_belief( clusters, obs, belief, prediction ):
  palette = generate_palette( len( clusters ) )
  pl.plot( obs[:,0], obs[:,1], "x", c = (0.7, 0.7, 0.7) )
  for n, t in enumerate( clusters ):
    t = np.array( t )
    for i in xrange( len( t ) ):
      pl.plot( t[i,0], t[i,1], "o", markersize = 15. * belief[n, i], c = palette[n] )
      pl.plot( t[i,0], t[i,1], "*", markersize = 60. * prediction[n, i], c = (1.0, 0.0, 0.0) )

def plot_clusters( e, data ):
  M = e.shape[1]
  palette = generate_palette( M )
  e_total = np.sum( e )
  for n, traj in enumerate( data ):
    traj = np.array( traj )
    c = np.array( [0., 0., 0.] )
    if np.sum( e[n, :] ) / e_total > 1E-4:
      e_m = np.sum( e[n, :] )
      for m in xrange( M ):
        c += palette[m] * e[n, m] / e_m
      pl.plot( traj[:,0], traj[:,1], "-", color = c )
    else:
      c = np.array( [0.8, 0.8, 0.8] )
      pl.plot( traj[:,0], traj[:,1], "-", color = c, alpha = 0.3 )



