# Letters classifier using Scikit-learn
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
            ├── emnist-balanced-mapping.txt
    ├── src
        └── letters_classifier.py
    └──....

__Note: The notebook emnist_overview.ipynb should not be modified.
    You should create new notebooks and re-do steps as in emnist_overview.ipynb__

## References
The *03_classification.ipynb* notebook that will guide you step by step through
digits classification project was included in notebooks directory.
This notebook is companion with the book *Hands on Machine Learning 3rd.e.d*.
You can also find it in: [Hands on ML 3's companion notebooks](https://github.com/ageron/handson-ml3)
