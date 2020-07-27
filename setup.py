import setuptools

from brainex.misc import pr_red, fd_workaround

with open("README.md", "r") as fh:
    long_description = fh.read()

USE_SPARK = True
try:
    print('Checking Spark backend')
    import pyspark
    sc = pyspark.SparkContext()
except:
    print('Either PySpark or Spark is not correclt installed. Skipping Spark packages.')
    print('You can still use Spark in Genex if you install Spark later.')
    USE_SPARK = False

requires = ['Cython',
            'numpy',
            'scipy',
            'pandas',
            'tslearn',
            'pyspark',
            ] if USE_SPARK else ['Cython',
                                 'numpy',
                                 'scipy',
                                 'pandas',
                                 'tslearn',
                                 ]

setuptools.setup(
    name="brainex",  # Replace with your own username
    version="0.2.5",
    author="ApocalyVec",
    author_email="s-vector.lee@hotmail.com",
    description="This package is a General Exploration System that implements DTW in exploring time series",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ApocalyVec/BrainEX",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    packages=setuptools.find_packages(),
    install_requires=requires)

fd_workaround()
