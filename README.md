# Genex 
General Exploration System that implements DTW in exploring time series.
This program implements the algorithm described in these papers:

http://real.mtak.hu/74287/1/p1595_neamtu_u.pdf
http://real.mtak.hu/43722/1/p169_neamtu_u.pdf
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8509275

## Recent Updates
* log level in do_gcluster, set the log level toggle on/off whether to spill out progression during clustering
[Jump to the section: Arguments for do_gcluster](#Arguments-for-do_gcluster)
* generate_query in parse
* gquery in gcluster


## Parse
Provides functions that help read csv files. 

### generate_source
Reads feature-time-series csv file.

    parse.generate_source(file_name, feature_num)
#### Arguments
* file_name: the name of the csv file to read
* feature_num: the number of components that describe a time series
#### Returns
a key-value pair list where key = (tuple) time series id, value is a list of raw data
#### Example

    from genex import parse
    fn = 'your_data.csv'
    # features: Subject Name, Event Name, Channel Name, Start time, End Time => total of five features
    res_list = parse.generate_source(fn, feature_num = 5)

### generate_query


## Preprocess
Preprocess input time series to prepare them for applying genex query
### do_gcluster
Pre-process the data by creating clusters.
    
    from genex import preprocess
    preprocess.do_gcluster(input_list: list, loi: tuple, sc: SparkContext, similarity_threshold: float = 0.1, dist_type: str='eu', normalize: bool=True, del_data: bool = False, data_slices:int=16, log_level:int=1)
#### Arguments for do_gcluster
* input_list: list of key-value pairs. It should be the output of [generate_source](#generate_source)
* loi: must be a list of two integers. length of interest
* sc: the spark context to which the do_gcluster job will be submitted
* similarity_threshold: must be a float between (0, 1), the similarity threshold defined in the Genex method. Generally speaking, larger threshold
produces fewer and larger cluster.  **Default at 0.1**

* dist_type: distance with which the cluster applies . It can be:
    * 'eu': Euclidean distance **Default**
    * 'ma': Manhattan (city-block) distance
    * 'mi': Minkowski distance
    * 'ch': chebyshev distance
* normalize: boolean whether to normalize the data in clustering calculations. It is general recommended to normalize
the data to help the performance. **Default True**
* del_data: boolean whether to delete the raw data from the sequence object created when clustering. It is generally 
recommened to delete the data after clustering to save memory space. The data can always be retrived by the 
fetch_data method of sequence **Default True**
* is_collect: boolean whether to collect the RDD as the final result. **Default True**
* data_slices: the number of slices with which the input data will be chopped up. The default value works well
on smaller data sets (# row < 20, average length < 400). If Pyspark starts to give **maximum task size warnings**. Try 
increase the number of slices (recommended to be a power of 2) **Default 16**
* log_level: integer value, if set to 1 (default), the program will print out the progress during clustering. **Default 1**

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
    from genex import parse
    from genex import preprocess

    # reads input data
    fn = 'your_data.csv'
    features_to_append = [0, 1, 2, 3, 4]
    res_list = parse.generate_source(fn, features_to_append)
    
    # initialize the spark context
    conf = SparkConf().setMaster("local").setAppName("Genex").set('spark.driver.memory', '16G')
    sc = SparkContext(conf=conf)
    clusters = preprocess.do_gcluster(input_list=res_list, loi=[50, 100], sc=sc, del_data=True)
# Gcluster
Return by genex.preprocess.do_gcluster. Gcluster is the general form that retains the Genex Cluster information.
## Attributes
* clusters(dict): dictionary object that holds the cluster information
    * key(integer): length of the sequence in the cluster
        * value(dict): the clustered sequence of keyed length
            * key(Sequence): Sequence object that is the representative
            * value(list): list of Sequence that are represented by the key
            
* filtered_data: dictionary object that has the same structre as the data. When returned by do_gcluster
filtered_data will have the same value as the data attribute. If filter is applied on the Gcluster object, the filtered
_data object will reflect that filter. See Gcluster.Methods.gfilter for more details
* data: the original data that is the input_list in [arguments for do_gcluster](#Arguments-for-do_gcluster)
* norm_data: a copy of the data attribute mentioned above, but the value points are normalized. The scale used to normalize
is saved to global_max and global_min in this class.
* global_max, global_min: the maximum and minimum value in the dataset. Set in do_gcluster. 
Those two attributes can be used to scale external query sequences on the same scale as the clusters

## Methods
Gcluster has various methods with which the user can retrieve and manipulate the Genex Cluster data.
### Collect
By default, in do_gcluster, isCollect is set to false. This means that if the user wish to view the cluster result
or calling any function that requires the cluster result such as len or slicing. The user must call collect() on the 
Gcluster first.
### Example
The following code will raise exception because Gcluster cannot be sliced if not collected. 
    
    from genex import preprocess

    clusters = do_gcluster(input_list=res_list, loi=[50, 100], sc=sc, del_data=True)
    clusters[75]

To be able to use slice or other functions that needs the cluster result, the user instead do the following
    
    from genex import preprocess

    clusters = do_gcluster(input_list=res_list, loi=[50, 100], sc=sc, del_data=True)
    clusters.collect()
    clusters[75]

Or have the result to be collected upon return by setting the isCollect parameter in do_gcluster
    
    from genex import preprocess

    clusters = do_gcluster(input_list=res_list, loi=[50, 100], sc=sc, del_data=True, isCollect=True)
    clusters[75]


### Slice
**Requires Gcluster to be collected.** 

Gcluster supports slicing to retrieve cluster data. Please not that Gcluster slice currently does NOT support stepping in slice.
#### Example

    from genex import preprocess

    clusters = do_gcluster(input_list=res_list, loi=[50, 100], sc=sc, del_data=False)
calling

    clusters[75]
will give the cluster of length 80 as a dictionary. The keys are the representatives and value are lists of Sequence
 that are represented by the keys

calling 

    clusters[75]
    clusters[75:]
    clusters[:75]
    clusters[50:75]
will give the list of cluster of the given slice.

### len
**Requires Gcluster to be collected.** 

Call  will return the number of cluster dictionaries.
#### Example

    from genex import preprocess

    clusters = do_gcluster(input_list=res_list, loi=[50, 100], sc=sc, del_data=False)
    print('len of Gcluster: ' + len(gcluster_obj))
will give    
    
    >>> len of Gcluster: 51
    
    
### Gcluster.gfilter
gfilter proves a comprehensive method with whcih the users may filter their clustered data by the sequence length and id.
After filtering, the filtered data will be stored in Gcluster.filtered_data, see Gclustter.attributes.filtered_data for
more details. Users may then proceed with the filtered_data to explore the clusters or implement custom queries. 

    Gcluster.gfilter(size=None, filter_features=None)
#### Arguments
gfilter can take up to two types of filter, size and feature. They are:
* size: integer or list or tuple, a list or tuple provided must has length of 2
    * if a integer is provided: any cluster whose length is not the given size will be filtered out.
    * if a list or a tuple is provided: the provided set will be treat as [start, end], where any cluster whose length is
    not in between start and end will be filtered out.
* filter_features: str or list or tuple
    * if a str is given: any sequence (other than the representatives) without the given feature str will be filtered out. 
    * if a list or a tuple is provided: similar to the str case, any sequence (other than the representatives) whose 
    features is not in the feature list will be filtered out.
    * **It is important to note that filter by feature will not affect the representatives in the clusters**
#### Returns
gfilter does not return the filter result, that filtered data is saved in the filtered_clusters attribute in Gcluster.
Calling gfilter more than once will overwrite previous filtered results.
#### Example
    from genex import preprocess

    clusters = do_gcluster(input_list=res_list, loi=[50, 100], sc=sc, del_data=False)
    
    # all the following are valid filter operations, B-DC4, A-DC8 are example features
    
    # Filter Example 1
    # only the representative and the sequence with feature 'B-DC4' wii be kept
    clusters.gfilter(filter_features ='B-DC4') 
    
    # Filter Example 2
    # first, only the clusters of length 50 and 51 will be kept, 
    # secondly, other than the representatives, sequences without id 'B-DC4' OR 'A-DC8' will be filtered out
    clusters.gfilter(size=(50, 51), filter_features = ['B-DC4', 'A-DC8']) 
    
    # Filter Example 2
    # all the clusters whose length is not 50 will be filtered out
    clusters.gfilter(size=50)
    
### Gcluster.get_representatives
get_representatives gives all the representative sequences in the clusters.
If filter is set to False: the method returns all the representative sequences.
If filter is et to True: the method only returns the representative in the filered_clusters attribute. 
To change the value of filtered_clusters see [Gcluster.gfilter](#Gcluster.gfilter)

    Gcluster.get_representatives(filter=False)
### Gcluster.get_cluster
get_cluster takes a representative sequence, and returns the cluster whom the given representative sequence represents.

    Gcluster.get_cluster(rep_seq):
####Arguments
rep_seq: a sequence object. The method will raise exception if the given sequence is not one of the representative. User
may use  [Gcluster.get_representatives](#Gcluster.get_representatives) to view the representative sequences.

#### Returns
get_cluster returns a list of sequences that is in the cluster whom the representative sequence represents.
### Gcluster.gquery
gquery is a step further from do_gcluster. It queries the clusters to find top k matches for a single query sequence.
Currently, gquery only supports query sequence from inside the original data. Next update will have it accept custom query
sequence.

    Gcluster.gquery(query_sequence: Sequence, sc: SparkContext,
                   loi=None, foi=None, k: int = 1, dist_type: str = 'eu', data_slices: int = 32,
                   ex_sameID: bool = False, overlap: float = 0.0)
#### Arguments
* query_sequence: a sequence object with which to query the clusters. Currently, gquery only supports query sequence from inside the original data. Next update will have it accept custom query
sequence.
* sc: the spark context on which the query operation will run
* loi: the length of interest for the query. Must be a integer or list or tuple, a list or tuple provided must has length of 2
    , only sequence whose size is within the length of interest will be queries.
* foi: the feature of interest. Must be a str or list or tuple, only sequence whose id contains the ANY of the given features will
be queries.
* k: integer value, gquery will find the best k matches.
* dist_type: distance with which the query applies . It can be:
    * 'eu': Euclidean distance **Default**
    * 'ma': Manhattan (city-block) distance
    * 'mi': Minkowski distance
    * 'ch': chebyshev distance
* data_slices: the number of slices with which the input data will be chopped up. The default value works well
on smaller data sets (# row < 20, average length < 400). If Pyspark starts to give **maximum task size warnings**. Try 
increase the number of slices (recommended to be a power of 2) **Default 32**
* ex_sameID: boolean. Whether to exclude sequences with the same id as the query sequence from the result.
* overlap: float value, must be between 0.0 and 1.0. The query result will exclude sequences whose mutual overlapping factor
exceeds the given value. 1 and 0
#### Returns
a list of key-value pairs. If k is greater than 1, the returning query result is sorted from closest match to the furthest.
value: the matching sequence
key: distance from the query sequence to the matching sequence
#### Example

    query_result = c.gquery(query_sequences[0],
                            sc=sc, loi=[50, 55],
                            foi=['B-DC4', 'A-DC8'],
                            k=3, dist_type='eu',
                            data_slices=16,
                            ex_sameID=False,
                            overlap=0.75)


## Query
### gquery
Genex query uses the result from [do_gcluster](#do_gcluster) as a base to match the sequence. It implemented as a class 
method in [Gcluster.gquery](#Gcluster.gquery)

### bfquery

## Bug Report
Please report any bug by creating issues.