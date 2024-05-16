import numpy as np
import pandas as pd
import geopandas as gpd
import os
from scipy.spatial.distance import cdist
from pyproj import Proj, transform

def load_block_shapes():
    """
    Args:
        state_abbrev: (str) two letter state abbreviation
        year: (int) the year of the TIGERLINE shapefiles

    Returns: (gpd.GeoDataFrame) of tract shapes
    """
    block_shapes = gpd.read_file('Robust/raw shapefiles')
    #block_shapes = block_shapes.to_crs("EPSG:4269")  # meters
    print(len(block_shapes))
    block_shapes = block_shapes[block_shapes.ALAND20 > 0]
    block_shapes = block_shapes[block_shapes.POP20 > 0]
    print(len(block_shapes))
    block_shapes=block_shapes.sort_values(by='GEOID20').reset_index(drop=True)
    print(block_shapes.NumEVs)
    block_df = pd.DataFrame({
        'GEOID': block_shapes.GEOID20.apply(lambda x: str(x).zfill(11)),
        'x': block_shapes.centroid.x,
        'y': block_shapes.centroid.y,
        'NumEVs1': block_shapes.NumEVs, #estimate based on households
        'NumEVs2': block_shapes.NumEVs2, #estimate based on population
        'Pop': block_shapes.POP20
    })

    block_df['NumEVs']=block_df[["NumEVs1", "NumEVs2"]].max(axis=1) #take the greater of the two estimates
    block_df=block_df.drop(['NumEVs1', 'NumEVs2'],axis=1)

    print(block_df)

    block_df.to_csv(os.path.join('Robust', 'block_df.csv'), index=False)

    return block_df

def load_chargers():
    chargers_df=pd.read_csv('Robust/chargers.csv')
    chargers_df['id2']=chargers_df['id']+chargers_df['angle']/1000
    chargers_df['prob']=chargers_df['prediction'] #probability of feasibility
    chargers_df['prediction']=np.round(chargers_df['prob']) #deterministic 0 or 1 estimate

    #transform lat lon
    crs_latlon = Proj(proj='latlong', datum='NAD83')
    crs_shapefile = Proj(init='epsg:4269')
    chargers_df['x'], chargers_df['y'] = transform(crs_latlon, crs_shapefile, chargers_df['longitude'].values, chargers_df['latitude'].values)

    print(np.sum(chargers_df['prediction']))
    print(chargers_df)

    chargers_df_05p=chargers_df[chargers_df.prob>=0.05]
    chargers_df_10p=chargers_df[chargers_df.prob>=0.1]
    chargers_df_02p=chargers_df[chargers_df.prob>=0.02]
    chargers_df_01p=chargers_df[chargers_df.prob>=0.01]

    chargers_df.to_csv(os.path.join('Robust', 'chargers_df.csv'), index=False)
    chargers_df_10p.to_csv(os.path.join('Robust', 'chargers_df_10p.csv'), index=False)
    chargers_df_02p.to_csv(os.path.join('Robust', 'chargers_df_02p.csv'), index=False)
    chargers_df_01p.to_csv(os.path.join('Robust', 'chargers_df_01p.csv'), index=False)

def generate_alpha(max_dist):
    #takes in the max dist in meters to be acceptably far away
    block_df=pd.read_csv('Robust/block_df.csv') 
    charger_df=pd.read_csv('Robust/chargers_df_10p.csv') #TODO

    block_points = block_df[['x', 'y']].values
    #print(block_points)
    charger_points = charger_df[['longitude', 'latitude']].values
    #print(charger_points)

    distances = cdist(block_points, charger_points)*111139 #multiply to convert to meters
    #print(distances)

    alpha=np.where(distances < max_dist, 1, 0)
    alpha=alpha.T
    print(np.sum(alpha, axis=0))

    #predictions=charger_df['prediction'].to_numpy()
    #print(predictions.shape)
    #print(alpha.shape)
    #alpha_pred=np.dot(alpha,predictions[:, np.newaxis])
    #print(alpha_pred)

    np.save('Robust/alpha%s_10p.npy' %max_dist, alpha)

def fix_block_results(B0):
    #results=pd.read_csv(os.path.join('Robust/preprocessed', 'satisfied_block_df_05p_300_%s.csv' %B0))
    results=pd.read_csv('Robust/preprocessed/satisfied_block_df_det_05p_300.csv')
    print(results)
    results['final_feas']=results['Finally feas']
    results.to_csv(os.path.join('Robust/preprocessed/satisfied_block_df_det_05p_300.csv'), index=False)

B0='092'
fix_block_results(B0)