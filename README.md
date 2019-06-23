# Genex 
General Exploration System that implements DTW in exploring time series.
This program implements the algorithm described in these papers:

http://real.mtak.hu/74287/1/p1595_neamtu_u.pdf
http://real.mtak.hu/43722/1/p169_neamtu_u.pdf
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8509275
# Parse
Provides functions that help read csv files. 

## generate_source
Reads feature-time-series csv file.

    generate_source(file_name, feature_num)
### Arguments
* file_name: the name of the csv file to read
* feature_num: the number of components that describe a time series
### Returns
a key-value pair list where key = (tuple) time series id, value is a list of raw data
### Example

    from genex.parse import generate_source
    fn = 'your_data.csv'
    # features: Subject Name, Event Name, Channel Name, Start time, End Time => total of five features
    res_list = generate_source(fn, feature_num = 5)

# Preproces
Preprocess input time series to prepare them for applying genex query
## gcluster
Pre-process the data by creating clusters.

    genex.preprocess.gcluster(input_list: list, loi: tuple, sc: SparkContext, similarity_threshold: float = 0.1, dist_type: str='eu', normalize: bool=True, del_data: bool = False, data_slices:int=16)
### Arguments
* input_list: list of key-value pairs
* loi: must be a list of two integers. length of interest
* sc: the spark context to which the gcluster job will be submitted
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

### Returns
cluster result as a list of dictionaries. Each dictionary is a sequence cluster of a certain length. The 
key is the representative sequence and the value is a list of sequences in that sequence and represented by the key sequence.
### Example
    from pyspark import SparkContext, SparkConf
    from genex.preprocess import gcluster
    from genex.parse import generate_source
    
    # reads input data
    fn = 'your_data.csv'
    features_to_append = [0, 1, 2, 3, 4]
    res_list = generate_source(fn, features_to_append)
    
    # initialize the spark context
    conf = SparkConf().setMaster("local").setAppName("Genex").set('spark.driver.memory', '16G')
    sc = SparkContext(conf=conf)
    clusters = gcluster(input_list=res_list, loi=[50, 100], sc=sc, del_data=True)

# Query
## gquery
## bfquery