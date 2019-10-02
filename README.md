# Genex
This repository is a General Exploration System that implements DTW in exploring time series.
This program implements the algorithm described in these papers:

Neamtu, Rodica, et al. "Interactive time series analytics powered by ONEX." Proceedings of the 2017 ACM International Conference on Management of Data. ACM, 2017.
Neamtu, Rodica, et al. "Generalized dynamic time warping: Unleashing the warping power hidden in point-wise distances." 2018 IEEE 34th International Conference on Data Engineering (ICDE). IEEE, 2018.
Stan Salvador, and Philip Chan. “FastDTW: Toward accurate dynamic time warping in linear time and space.” Intelligent Data Analysis 11.5 (2007): 561-580.

In addition, Genex uses Spark as distributed computing engine, whose reference can be found [here](https://spark.apache.org/docs/latest/)


## Genex Database
Genex database (Aliase: gxdb) is core object in genex. It keeps the original time series given by the user in addition to some meta-data.
Note that gxdb must exist in a [Spark Context](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#module-pyspark.sql).

## Creat Genex Database
There are two ways to create a gxdb in a given context. [from_csv] and [from_db]


## API Calls
### from_csv
```
from_csv(file_name, feature_num, sc)
```
returns a gxdb object from given csv data file;  the returned gxdb
resides in the SparkContext given.
#### Arguments
**file_name**: The input file name given by the user. File format

| Feature1 | Feature2 | Feature3 | Feature4 | Feature5 |        |        |     |        |
|----------|----------|----------|----------|----------|--------|--------|-----|--------|
| val11    | val21    | val31    | val41    | val51    | data11 | data12 | ... | data1n |
| val12    | val22    | val32    | val42    | val52    | data21 | data22 | ... | data2n |
| val13    | val23    | val33    | val43    | val53    | data31 | data32 | ... | data3n |


**feature_num**: The number of features in the csv file. Eg: if the features are Subject Name, Event Name, Channel Name, Start time, End Time, then the number of features is 5.
**sc**: Spark context instance, must be created in the Python context. Note there be only be one gxdb instance on a single spark context.
#### Exceptions
FileNotFoundError: If the file can not be found in given path, raise error.
TypeError: If the input type is not same as required input type, raise error.

#### Example
```
from genex.database import genex_database as gxdb

mydb = from_csv(‘mydata.csv’, feature_num=5)
```

### from_db
```
from_db(path, sc)
```
A ’load back’ function, returns a **previously saved** gxdb object from its saved path; the returned gxdb resides in the SparkContext given. The reload SparkContext for the gxdb does not have to be the same as the context where the gxdb is created.
#### Arguments
**path**
**sc**
#### Exceptions
FileNotFoundError
#### Example

### genex_database.build
Groups and clusters the time series set based on the customized similarity threshold and the distance type which is a parameter in the DTW algorithm to calculate the similarity between two time series.

Args：
- similarity_threshold: The upper bound of the similarity value between two time series.
- dist_type: 'eu'
- loi: Length of interest, it is a slice object.
- verbose:


**save**
```
save(
   folder_name: str
) -> None
```
Stores a gxdb instance locally which can be fetched and restored in the future.

Args:
- folder_name: Give a name to the folder, which stores all the essential fields of the gxdb instance


**from_db**
```
from_db（
   sc: SparkContext,
   folder_name: str
）-> genex_database
```
Creates an new gxdb based on an existed one which was stored in the past.

Args:
- sc: A instacne of the SparkContext.
- folder_name: The folder name which contains the local gxdb instance.
  
  
**from_csv**
```
from_csv(
   file_name: str,
   feature_num: int,
   SparkContext: SparkContext
) -> genex_database

```
Creates an new gxdb based on the given csv file.

Args:
- file_name: A csv file that contains time series values
- feature_num：An ID used to distinguish each time series
- SparkContext: A instance of SparkContext


**query**
```
query(
   query: Sequence,
   best_k: int,
   unique_id: bool,
   overlap: float
) -> pandas dataFrame
```
Find the k-best similar time series based on the given query time series from the current gxdb instance

Args:
- query: The query time series.
- best_k: k best matches.
- unique_id: A boolean value that determinate whether the query results are allowed to have the same ID
- overlap: Sets up the upper bound of the overlap among the query candidates.


mydb=from_csv(...)
mydb.save(path)

-----

mydb=from_db()
