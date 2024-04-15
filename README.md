# Running the experiments

1. Edit the `EXPERIMENT CONFIGURATION` section of the `analysis.py` file to set the appropriate experimental setup.
2. Run `python3 analysis.py configuration_experiments [experiment_name]` script (the first argument is the name of the file with the selected configuration, and the second argument, optional, is an additional name for the experiment logs folder).

If you do not want that python uses buffered output, which is useful when you want to see stdout logs as soon as they are produced, especially when the stdout is written to a file (e.g. nohup), where large buffers are used that may retain the output for a while, run python with `-u` option (unbuffered).

For example, `python3 -u analysis.py configuration_experiments KNN`


The models developed in this study are:
<ul>
  <li><strong>PPIIBM_first_item</strong>, Pair Prediction by Item Identification Baseline Model (first item mode)</li>
  <li><strong>PPIIBM_both_items</strong>, Pair Prediction by Item Identification Baseline Model (both items mode)</li>
</ul> 

The classic machine learning models that can be selected are:
<ul>
  <li><strong>KNN</strong>, k-nearest neighbors</li>
  <li><strong>LR</strong>, logistic regression classifier</li>
  <li><strong>RF</strong>, random forest classifier</li>
  <li><strong>SVC</strong>, Support Vector Classifier</li>
</ul>



# Creating the virtual environment

## Python venv

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

# Running with GPU
It is necessary to have Conda previously installed on your system.

Creating the Conda environment with RAPIDS:

```
conda create -n rapids-24.02 -c rapidsai -c conda-forge -c nvidia cuml=24.02 python=3.10 cuda-version=11.8
```

Once the environment is activated, you can run the Python scripts to use RAPIDS and execute them on GPU by changing the `use_GPU = True` flag in the experiment configuration file.