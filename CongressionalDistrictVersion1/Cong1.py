# -*- coding: utf-8 -*-

"""
This file tests a slightly different chain to CHAIN_TEST1 (we also need to test different starting partitions)
Here I've simply removed the compactness requirement and only recorded every 10 steps (but gone up to 10000 total steps to record same data)
"""

import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import maup
import numpy as np
import time
import networkx as nx
import gerrychain
import math
from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                        proposals, updaters, constraints, accept, Election)
from gerrychain.proposals import recom
import gerrychain.updaters
from gerrychain.metrics import efficiency_gap, mean_median
from gerrychain.updaters import cut_edges, Tally

from functools import partial

#Name which columns we should use for population and as the block id
popCol="tot" #tot19 isn't a valid partition since this map was made with 2010 data
unique_label="GEOID10"
county_label="COUNTYFP10"
typeOfElection="CongDist"

#______LOAD DATA__________
graph=gerrychain.Graph.from_json("../MI_graph.json")
#Here we import the coi data and the (cleaned) michigan data
MI_rep_data = gpd.read_file("../MI_COIS.geojson")#,crs="EPSG:4326")
MI_rep_data=MI_rep_data.to_crs("EPSG:6493")

michigan_data=gpd.read_file("../BlockData/MI_blocks_all_data_full.shp")#, crs="EPSG:6493")
michigan_data=michigan_data.to_crs("EPSG:6493")



#____CLEAN DATA______
michigan_data[popCol]=michigan_data[popCol].fillna(0) #if we are using tot19 its important to remove nas
michigan_data=michigan_data.reset_index()
MI_rep_data=MI_rep_data.reset_index()
michigan_data.drop("index",axis=1)
MI_rep_data.drop("index",axis=1)




#_____ASSOCIATE COIs WITH PRECINCTS________
COIs=[]
for index,row in MI_rep_data.iterrows():
    tempCOIgeom=MI_rep_data.loc[index,"geometry"]
    intersectingPrecincts=michigan_data.intersects(tempCOIgeom,align=False)
    
    #Instead of storing a 1 or a 0, store proportion (nicer for some metrics and can always check >0 easily)
    #Assume everyone in precinct belongs to COI (which is a bad assumption that we should change )
    COI_pop=sum(list(michigan_data[intersectingPrecincts][popCol]))

    michigan_data["COI"+str(index)]=np.multiply(intersectingPrecincts,np.array(list(michigan_data[popCol])))/COI_pop #can't use name because some of these may have the same name (could also just change by hand).
    COIs.append(MI_rep_data.loc[index,"entry_name"])
    

#Output a nice image just to show where our COIs are
fig,ax=plt.subplots(1,1,figsize=(10,10))
michigan_data.plot(ax=ax);
MI_rep_data.plot(column="entry_name",ax=ax,alpha=.7);
plt.suptitle("All COI Map")
plt.savefig("All COI Map")

print("COI Labels")
for i in range(len(COIs)):
    print(i,COIs[i])
    

#__________DEFINE METRICS ___________
def get_current_assignment_toindex(partition):
    return [partition.assignment[gid] for gid in range(len(michigan_data[unique_label]))]

def num_COI_splits(partition,COI_LIST):
    #Count districts with at least 1 precinct in COI. Subtract 1 (since if all COI precincts are in the same district, then we have no splits)
    michigan_data["current"]=get_current_assignment_toindex(partition)
    result={}
    groupedData=(michigan_data.groupby(["current"]).sum())
    for COI in COI_LIST:
        colForCOI=groupedData[COI]
        result[COI]=colForCOI.to_numpy().nonzero()-1 
    return result

def UncertainityOfMembership(partition,COI_LIST):
    michigan_data["current"]=get_current_assignment_toindex(partition)
    result={}
    groupedData=(michigan_data.groupby(["current"]).sum())
    for COI in COI_LIST:
        colForCOI=groupedData[COI]
        result[COI]=sum([-colForCOI[indT]*math.log(colForCOI[indT],2) if colForCOI[indT]>0 else 0 for indT in range(len(colForCOI))])
    return result

def num_county_splits(partition):
    michigan_data["current"]=get_current_assignment_toindex(partition)
    splits = sum(
        michigan_data.groupby(county_label)["current"].nunique() > 1)
    return splits

#__________SET UP GERRYCHAIN_______

num_elections = 1
election_names = ["PRES16"]  #This is what the shapefile came with, we may want something else
election_columns = [["PRES16D", "PRES16R"]]
updaters = {
    "population": updaters.Tally(popCol, alias="population"),
    "cut_edges": cut_edges,
}
elections = [
    Election(
        election_names[i],
        {"Democratic": election_columns[i][0], "Republican": election_columns[i][1]},
    )
    for i in range(num_elections)
]
election_updaters = {election.name: election for election in elections}
updaters.update(election_updaters)


initial_partition = Partition(graph, typeOfElection, updaters)
ideal_population = sum(initial_partition["population"].values()) / len(
    initial_partition
)

fig,ax=plt.subplots(figsize=(10,10))
print("Ideal population: ", ideal_population)
plt.suptitle("State House Districts in Michigan")
initial_partition.plot(michigan_data,ax=ax)
plt.savefig("InitialCongMap")



proposal = partial(
    recom, pop_col=popCol, pop_target=ideal_population, epsilon=0.0005, node_repeats=2
)
compactness_bound = constraints.UpperBound(
    lambda p: len(p["cut_edges"]), 2 * len(initial_partition["cut_edges"])
)
chain = MarkovChain(
    proposal=proposal,
    constraints=[
        constraints.within_percent_of_ideal_population(initial_partition, 0.0005),
        compactness_bound,  
        # single_flip_contiguous#no_more_discontiguous
    ],
    accept=accept.always_accept,
    initial_state=initial_partition,
    total_steps=1000,
)

#_____RUN THE CHAIN_________
import warnings
warnings.filterwarnings("ignore") #nan in some fields still
i=0

burnInTime=0
timeBetweenSamples=1
start_time=time.time()

partition_info=pd.DataFrame(columns=["Step",
                                     #"StrPartition", I was going to stringify the output for future reference but this will be memory instensive
                                     "county_splits",
                                     "pop_deviation",
                                     "cut_edges",
                                     "Mean-median",
                                     "Efficiency_gap",
                                     "Dem_Victories",
                                    ]+["COI"+str(i)+"_UoM" for i in range(len(COIs))]
)
start_time=time.time()
for part in chain: 
    #After first five iterations print out the time taken so I have a speed to report
    if(i==5*timeBetweenSamples):    
        print(i,(time.time()-start_time)/5,"s")
    
    if(i<burnInTime and i>0):
        #always record the info for the initial partition but then skip burn time
        pass
    
    elif(i%timeBetweenSamples==0):
        COI_UMS=UncertainityOfMembership(part,["COI"+str(i) for i in range(len(COIs))])
        partition_info.loc[len(partition_info.index)] = [i, 
                             num_county_splits(part),
                            #sorted(list(part["population"].values())), #Should we save this data or just hold onto a few metric?
                            (max(part["population"].values())-min(part["population"].values()))/ideal_population, #Pop Deviation
                             len(part["cut_edges"]), #cut edges
                             mean_median(part[election_names[0]]), #mean-median gap
                             efficiency_gap(part[election_names[0]]), #efficiency gap
                             part[election_names[0]].wins("Democratic"), #number of democratic wins       
                            ]+[COI_UMS["COI"+str(i)] for i in range(len(COIs))]  #number of splits in each COI 
    if(i%50==0):
        fig,ax=plt.subplots(figsize=(10,10))
        print("Ideal population: ", ideal_population)
        plt.suptitle("State House Districts in Michigan")
        part.plot(michigan_data,ax=ax)
        plt.savefig(str(i)+"_Step_Partitition")
    i+=1
    

partition_info.to_csv("Cong1.csv")




