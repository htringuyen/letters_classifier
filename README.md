# EMNIST Letter classification using Scikit-learn

## Course EE3021 - Semester 222: Report files
1. The report pdf file path: letters_classifier/emnist_final_report.pdf
2. The source code path: letters_classifier/src
3. The jupyter notebooks path: letters_classifier/notebooks
4. The data used in this project is in: letters_classifier/data

## Setup working project
Create Pycharm scientific project and config conda environment.

The data and source code must be structured as follows:

    project-name
    ├── data
    │   ├── emnist_gz
            ├── eminist-balanced-train-images-idx3-ubyte.gz
            ├── eminist-balanced-train-labels-idx1-ubyte.gz
            ├── eminist-balanced-test-images-idx3-ubyte.gz
            ├── eminist-balanced-test-labels-idx1-ubyte.gz
    │   └── mapping
            └── emnist-balanced-mapping.txt
    ├── src
    │   ├── emnist_utils.py
        └── datacontainer.py
    └──....

__Note: The notebook emnist_overview.ipynb should not be modified.
    You should create new notebooks and re-do steps as in emnist_overview.ipynb__

## References
The *03_classification.ipynb* notebook that will guide you step by step through
digits classification project was included in notebooks directory.
This notebook is companion with the book *Hands on Machine Learning 3rd.e.d*.
You can also find it in: [Hands on ML 3's companion notebooks](https://github.com/ageron/handson-ml3)