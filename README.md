# Genex 
General Exploration System that implements DTW in exploring time series.
This program implements the algorithm described in these papers:

http://real.mtak.hu/74287/1/p1595_neamtu_u.pdf
http://real.mtak.hu/43722/1/p169_neamtu_u.pdf
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8509275
## Parse
Provides functions that help read csv files. 

### generate_source
Reads feature-time-series csv file.

    generate_source(file_name, feature_num)
#### Arguments
* file_name: the name of the csv file to read
* feature_num: the number of components that describe a time series
#### Returns
a key-value pair list where key = (tuple) time series id, value is a list of raw data
#### Example

    from genex.parse import generate_source
    fn = 'your_data.csv'
    # features: Subject Name, Event Name, Channel Name, Start time, End Time => total of five features
    res_list = generate_source(fn, feature_num = 5)

## Preproces
Preprocess input time series to prepare them for applying genex query
### do_gcluster
Pre-process the data by creating clusters.

    genex.preprocess.do_gcluster(input_list: list, loi: tuple, sc: SparkContext, similarity_threshold: float = 0.1, dist_type: str='eu', normalize: bool=True, del_data: bool = False, data_slices:int=16)
#### Arguments
* input_list: list of key-value pairs
* loi: must be a list of two integers. length of interest
* sc: the spark context to which the do_gcluster job will be submitted
* similarity_threshold: must be a float between (0, 1), the similarity threshold defined in the Genex method. Generally speaking, larger threshold
produces fewer and larger cluster.  **Default at 0.1**

* dist_type: distance with which the cluster applies . It can be:
    * 'eu': Euclidean distance **Default**
    * 'ma': Manhattan (city-block) distance
    * 'mi': Minkowski distance
* normalize: boolean whether to normalize the data in clustering calculations. It is general recommended to normalize
the data to help the performance. **Default True**
* del_data: boolean whether to delete the raw data from the sequence object created when clustering. It is generally 
recommened to delete the data after clustering to save memory space. The data can always be retrived by the 
fetch_data method of sequence **Default True**
* is_collect: boolean whether to collect the RDD as the final result. **Default False**
    

#### Returns
Depending on Gcluster object that holds the cluster result.
* if set to false (default): the return will still be a gcluster, but its result is reversed for further process, refer
to Collect in Gcluster for more details.
* if set to true: the gluster is already collected upon return. The user may view the clusters and call functions that 
needs the cluster result such as len or slicing. 
cluster result as a list of dictionaries. Each dictionary is a sequence cluster of a certain length. The 
key is the representative sequence and the value is a list of sequences in that sequence and represented by the key sequence.
#### Example
    from pyspark import SparkContext, SparkConf
    from genex.preprocess import do_gcluster
    from genex.parse import generate_source
    
    # reads input data
    fn = 'your_data.csv'
    features_to_append = [0, 1, 2, 3, 4]
    res_list = generate_source(fn, features_to_append)
    
    # initialize the spark context
    conf = SparkConf().setMaster("local").setAppName("Genex").set('spark.driver.memory', '16G')
    sc = SparkContext(conf=conf)
    clusters = do_gcluster(input_list=res_list, loi=[50, 100], sc=sc, del_data=True)
## Gcluster
RReturn by genex.preprocess.do_gcluster. Gcluster is the general form that retains the Genex Cluster information.
### Attributes
* data_dict(dict): dictionary object that holds the cluster information
    * key(integer): length of the sequence in the cluster
        * value(dict): the clustered sequence of keyed length
            * key(Sequence): Sequence object that is the representative
            * value(list): list of Sequence that are represented by the key
### Methods
Gcluster has various methods with which the user can retrieve and manipulate the Genex Cluster data.
#### Collect
By default, in do_gcluster, isCollect is set to false. This means that if the user wish to view the cluster result
or calling any function that requires the cluster result such as len or slicing. The user must call collect() on the 
Gcluster first.
#### Example
The following code will raise exception because Gcluster cannot be sliced if not collected. 

    clusters = do_gcluster(input_list=res_list, loi=[50, 100], sc=sc, del_data=True)
    clusters[75]

To be able to use slice or other functions that needs the cluster result, the user instead do the following
    
    clusters = do_gcluster(input_list=res_list, loi=[50, 100], sc=sc, del_data=True)
    clusters.collect()
    clusters[75]

Or have the result to be collected upon return by setting the isCollect parameter in do_gcluster
    
    clusters = do_gcluster(input_list=res_list, loi=[50, 100], sc=sc, del_data=True, isCollect=True)
    clusters[75]


#### Slice
**Requires Gcluster to be collected.** 

Gcluster supports slicing to retrieve cluster data. Please not that Gcluster slice currently does NOT support stepping in slice.
##### Example
    gcluster_obj = do_gcluster(input_list=res_list, loi=[50, 100], sc=sc, del_data=False)
calling

    gcluster_obj[75]
will give the cluster of length 80 as a dictionary. The keys are the representatives and value are lists of Sequence
 that are represented by the keys

calling 

    gcluster_obj[75]
    gcluster_obj[75:]
    gcluster_obj[:75]
    gcluster_obj[50:75]
will give the list of cluster of the given slice.

#### len
**Requires Gcluster to be collected.** 

Call  will return the number of cluster dictionaries.
##### Example
    gcluster_obj = do_gcluster(input_list=res_list, loi=[50, 100], sc=sc, del_data=False)
    print('len of Gcluster: ' + len(gcluster_obj))
will give    
    
    >>> len of Gcluster: 51
    
    
#### get_clusters

## Query
### gquery
### bfquery

## Bug Report
Please report any bug by creating issues.