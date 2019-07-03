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

    parse.generate_source(file_name, feature_num)
#### Arguments
* file_name: the name of the csv file to read
* feature_num: the number of components that describe a time series
#### Returns
a key-value pair list where key = (tuple) time series id, value is a list of raw data
#### Example

    from Genex import parse
    fn = 'your_data.csv'
    # features: Subject Name, Event Name, Channel Name, Start time, End Time => total of five features
    res_list = parse.generate_source(fn, feature_num = 5)

## Preproces
Preprocess input time series to prepare them for applying genex query
### do_gcluster
Pre-process the data by creating clusters.
    
    from Genex import preprocess
    preprocess.do_gcluster(input_list: list, loi: tuple, sc: SparkContext, similarity_threshold: float = 0.1, dist_type: str='eu', normalize: bool=True, del_data: bool = False, data_slices:int=16)
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
    from Genex import parse
    from Genex import preprocess

    # reads input data
    fn = 'your_data.csv'
    features_to_append = [0, 1, 2, 3, 4]
    res_list = parse.generate_source(fn, features_to_append)
    
    # initialize the spark context
    conf = SparkConf().setMaster("local").setAppName("Genex").set('spark.driver.memory', '16G')
    sc = SparkContext(conf=conf)
    clusters = preprocess.do_gcluster(input_list=res_list, loi=[50, 100], sc=sc, del_data=True)
## Gcluster
RReturn by genex.preprocess.do_gcluster. Gcluster is the general form that retains the Genex Cluster information.
### Attributes
* data(dict): dictionary object that holds the cluster information
    * key(integer): length of the sequence in the cluster
        * value(dict): the clustered sequence of keyed length
            * key(Sequence): Sequence object that is the representative
            * value(list): list of Sequence that are represented by the key
            
* filtered_data: dictionary object that has the same structre as the data. When returned by do_gcluster
filtered_data will have the same value as the data attribute. If filter is applied on the Gcluster object, the filtered
_data object will reflect that filter. See Gcluster.Methods.gfilter for more details
### Methods
Gcluster has various methods with which the user can retrieve and manipulate the Genex Cluster data.
#### Collect
By default, in do_gcluster, isCollect is set to false. This means that if the user wish to view the cluster result
or calling any function that requires the cluster result such as len or slicing. The user must call collect() on the 
Gcluster first.
#### Example
The following code will raise exception because Gcluster cannot be sliced if not collected. 
    
    from Genex import preprocess

    clusters = do_gcluster(input_list=res_list, loi=[50, 100], sc=sc, del_data=True)
    clusters[75]

To be able to use slice or other functions that needs the cluster result, the user instead do the following
    
    from Genex import preprocess

    clusters = do_gcluster(input_list=res_list, loi=[50, 100], sc=sc, del_data=True)
    clusters.collect()
    clusters[75]

Or have the result to be collected upon return by setting the isCollect parameter in do_gcluster
    
    from Genex import preprocess

    clusters = do_gcluster(input_list=res_list, loi=[50, 100], sc=sc, del_data=True, isCollect=True)
    clusters[75]


#### Slice
**Requires Gcluster to be collected.** 

Gcluster supports slicing to retrieve cluster data. Please not that Gcluster slice currently does NOT support stepping in slice.
##### Example

    from Genex import preprocess

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

    from Genex import preprocess

    gcluster_obj = do_gcluster(input_list=res_list, loi=[50, 100], sc=sc, del_data=False)
    print('len of Gcluster: ' + len(gcluster_obj))
will give    
    
    >>> len of Gcluster: 51
    
    
#### gfilter
gfilter proves a comprehensive method with whcih the users may filter their clustered data by the sequence length and id.
After filtering, the filtered data will be stored in Gcluster.filtered_data, see Gclustter.attributes.filtered_data for
more details. Users may then proceed with the filtered_data to explore the clusters or implement custom queries. 
    Gcluster.gfilter(self, size=None, filter_features=None):
##### Arguments
gfilter can take up to two types of filter, size and feature. They are:
* size: integer or list or tuple, a list or tuple provided must has length of 2
    * if a integer is provided: any cluster whose length is not the given size will be filtered out.
    * if a list or a tuple is provided: the provided set will be treat as [start, end], where any cluster whose length is
    not in between start and end will be filtered out.
* filter_features: str or list or tuple
    * if a str is given: any sequence (other than the representatives) without the given feature str will be filtered out. 
    * if a list or a tuple is provided: similar to the str case, any sequence (other than the representatives) whose 
    features is not in the feature list will be filtered out.
    * ** It is important to note that filter by feature will not affect the representatives in the clusters**
##### Example
    from Genex import preprocess

    gcluster_obj = do_gcluster(input_list=res_list, loi=[50, 100], sc=sc, del_data=False)
    
    # all the following are valid filter operations, B-DC4, A-DC8 are example features
    
    # Filter Example 1
    # only the representative and the sequence with feature 'B-DC4' wii be kept
    gcluster_obj.gfilter(filter_features = ='B-DC4') 
    
    # Filter Example 2
    # first, only the clusters of length 50 and 51 will be kept, 
    # secondly, other than the representatives, sequences without id 'B-DC4' OR 'A-DC8' will be filtered out
    gcluster_obj.gfilter(size=(50, 51), filter_features = ['B-DC4', 'A-DC8']) 
    
    # Filter Example 2
    # all the clusters whose length is not 50 will be filtered out
    gcluster_obj.gfilter(size=50)


#### get_feature_list
if size is a list or a tuple, the first number must be smaller than the second

#### get_representatives


## Visualize

## Query
### gquery
### bfquery

## Bug Report
Please report any bug by creating issues.