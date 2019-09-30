## genex.database

Aliase: gxdb

### Class genex_database

\_\_init__

```
__init__(**kwargs)
```
Creates a new gxdb instance, which sets up 4 fields at first--data, data_normalized, scaler and sc.
Note that we will not create a gxdb instance directly through the initializer.
#### Methods

build
```
build(
    similarity_threshold: float,
    dist_type: str,
    loi：Slice object
    verbose: int
) -> None
```
Groups and clusters the time series set based on the customized similarity threshold and the distance type which is a parameter
in the DTW algorithm to calculate the similarity between two time series.

Args：
     - similarity_threshold: The upper bound of the similarity value between two time series.
     - dist_type: 'eu'
     - loi: Length of interest, it is a slice object.
     - verbose:

save
```
save(
    folder_name: str
) -> None
```
Stores a gxdb instance locally which can be fetched and restored in the future. 

Args:
    - folder_name: Give a name to the folder, which stores all the essential fields of the gxdb instance

from_db
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
    
from_csv
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

query
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
