import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="genex-ApocalyVec", # Replace with your own username
    version="0.0.1",
    author="ApocalyVec",
    author_email="s-vector.lee@hotmail.com",
    description="This package is a General Exploration System that implements DTW in exploring time series",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ApocalyVec/Genex",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
