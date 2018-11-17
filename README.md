![travis ci](https://travis-ci.org/usc-isi-i2/dsbox-ta2.svg?branch=master)

# DSBox: Automated Machine Learning System #

## Installation Instructions ##

### Installing DSBox using base D3M Image ###

Get docker image from:

```
docker pull registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-artful-python36-v2018.7.10-20180801-215033
```

Start the container:

```
docker run --name isi-dsbox -it 48a23667534e /bin/bash
```

And, within the container do the following:

Create the directories:

```
mkdir /output
mkdir /input
mkdir -p /user_opt/dsbox
```

Installing DSBox software and install SKLearn and D3M common primitves:

```
cd /user_opt/dsbox

git clone https://github.com/usc-isi-i2/dsbox-ta2.git --branch eval-2018-summer
cp dsbox-ta2/d3mStart.sh /user_opt/
chmod a+x /user_opt/d3mStart.sh

pip3 install -e git+https://gitlab.com/datadrivendiscovery/sklearn-wrap@9346c271559fd221dea4bc99c352ce10e518759c#egg=sklearn-wrap
pip3 install -e git+https://gitlab.com/datadrivendiscovery/common-primitives@fa865a1babc190055cb2a17cbdcd5b37e6f5e774#egg=common-primitives
```

### Using Exisitng DSBox Docker Image ###

A pre-build DSBox Docker image is here: [DSBox TA2 Image for 2018 Summer Evaluation](https://hub.docker.com/r/uscisii2/dsbox/)

Notes on how to run DSBox TA2 on non-D3M datasets is here: [Running DSBox TA2 on Other Datasets](https://github.com/usc-isi-i2/dsbox-ta2-system/blob/master/docker/dsbox_train_test/run_dsbox_with_other_dataset.md)


## Running DSBox ##

### Running DSBox search using the D3M submission evaluation script `d3mstart.sh` ###

The script that we provided from evaluation is `/user_opt/d3mStart.sh`. This script expects several shell environmental variables to be set. These shell environmental variables are defined here:  [2018 Summer Evaluation - Execution Process](https://datadrivendiscovery.org/wiki/display/gov/2018+Summer+Evaluation+-+Execution+Process)

The shell environmental variable `D3MINPUTDIR` defines the directory containing the `search_config.json` file that defines the dataset and problem to be solved. The format of this json configuration file is defined here: [JSON Configuration File Format](https://datadrivendiscovery.org/wiki/pages/viewpage.actionpageId=11275766)

Below is a sample `search_config.json` file, see `/user_opt/dsbox/dsbox-ta2/dataset/38_sick/search_config.json`. All the pipelines generated during the search process is stored in the `/output/38_sick/pipelines` directory. The pipeline "executables" are stored in the `/output/38_sick/executables` directory, and the actual pickled primitives of the pipeline executables are stored in the `/output/38_sick/supporting_files` directory.
```javascript
{
    "problem_schema": "/user_opt/dsbox/dsbox-ta2/dataset/38_sick/38_sick_problem_TRAIN/problemDoc.json",
    "problem_root": "/user_opt/dsbox/dsbox-ta2/dataset/38_sick/38_sick_problem_TRAIN",
    "dataset_schema": "/user_opt/dsbox/dsbox-ta2/dataset/38_sick/38_sick_dataset_TRAIN/datasetDoc.json",
    "training_data_root": "/user_opt/dsbox/dsbox-ta2/dataset/38_sick/38_sick_dataset_TRAIN",
    "pipeline_logs_root": "/output/38_sick/pipelines",
    "executables_root": "/output/38_sick/executables",
    "user_problems_root": "/output/38_sick/user_problems",
    "temp_storage_root": "/output/38_sick/supporting_files"
}

```

To run the DSBox TA2 search on the provided sample dataset 38_sick in `/user_opt/dsbox/dataset/38_sick/search_config.json`:

```
export D3RUN=search
export D3MOUTPUTDIR=/output/38_sick
export D3MINPUTDIR=/user_opt/dsbox/dsbox-ta2/dataset/38_sick
export D3MTIMEOUT=60
export D3MCPU=8
export D3MRAM=16Gi

/user_opt/d3mStart.sh
```

### Running DSBox search using the `ta2-search` script ###

Also, for convience we provide second script to run DSBox. This script accepts commandline arguments, instead of accessing shell environmental variables.

To see the command line options, do:

```
python /user_opt/dsbox/dsbox-ta2/python/ta2-search --help
```

For example, to run the sample `38_sick` problem with a 10-minute limit do:

```
python /user_opt/dsbox/dsbox-ta2/python/ta2-search --timeout 10 /user_opt/dsbox/dsbox-ta2/dataset/38_sick
```

### Running a Single DSBox Template ###

To search using a single template use the `ta1-run-single-template` shell script. See the next sections for an explanation of DSBox templates.

```
python /user_opt/dsbox/dsbox-ta2/python/ta1-run-single-template --template default_classification_template --timeout 10 /user_opt/dsbox/dsbox-ta2/dataset/38_sick
```

### Running a fitted pipeline on a test dataset ###

The fitted pipelines generated by the DSBox system are stored in the directory defined by the `executables_root` in the `search_config.json` file. The name of DSBox fitted pipelines have the form `<UUID>.json`. To run a fitted pipeline, say `0ecd7d2e-4620-430a-881e-b93ff0cfb692.json`, on the `38_sick` problem, do:

First create a `test_config.json` file, see `/user_opt/dsbox/dsbox-ta2/dataset/38_sick/test_config.json`. The format of this json configuration file is defined here: [JSON Configuration File Format](https://datadrivendiscovery.org/wiki/pages/viewpage.actionpageId=11275766)

```javascript
{
    "problem_schema": "/user_opt/dsbox/dsbox-ta2/dataset/38_sick/38_sick_problem_TEST/problemDoc.json",
    "problem_root": "/user_opt/dsbox/dsbox-ta2/dataset/38_sick/38_sick_problem_TEST",
    "dataset_schema": "/user_opt/dsbox/dsbox-ta2/dataset/38_sick/38_sick_dataset_TEST/datasetDoc.json",
    "test_data_root": "/user_opt/dsbox/dsbox-ta2/dataset/38_sick/38_sick_dataset_TEST",
    "results_root": "/output/38_sick/predictions",
    "executables_root": "/output/38_sick/executables",
    "temp_storage_root": "/output/38_sick/supporting_files"
}
```

Then run the fitted pipeline `0ecd7d2e-4620-430a-881e-b93ff0cfb692.json`:


```
export D3RUN=test
export D3MTESTOPT=0ecd7d2e-4620-430a-881e-b93ff0cfb692.json
export D3MOUTPUTDIR=/output/38_sick
export D3MINPUTDIR=/user_opt/dsbox/dsbox-ta2/dataset/38_sick
export D3MTIMEOUT=60
export D3MCPU=8
export D3MRAM=16Gi

/user_opt/d3mStart.sh
```

## Customizing DSBox Search Space ##

DSBox organizes the search space of possible machine learning pipelines into a collection
of pipeline templates, where each template implicilty defines configuration search
space. The set of templates is maintained by the Python class
`dsbox.template.library.TemplateLibrary`.

Here is a paritial list of templates in the template library:

* Default Classification Template
* Default Regression Template
* SRI Mean Baseline Template
* Default Timeseries Regression Template
* Default Text Regression Template
* Default Image Processing Regression Template
* Naive Bayes Classification Template
* Default Text Classification Template


To modify the DSBox search space, one can:

* Add/Remove templates from the TemplateLibrary
* Add/Remove primitives from template steps
* Augment the hyperparameter search space of each primitive


### DSBox Pipeline Templates ###

DSBox template is similar to the D3M concept of pipeline. A D3M
pipeline is a directed acyclic graph where the nodes are machine learning primitives and
the edges specify the order of computation and data dependency. A D3M pipeline node is
bounded to a single primitive, whereas a DSBox template node can be bounded to multiple
primitives. In addition, DSBox allows a hyperparameter search space to be associated with
each primitive. Each template defines a configuration space to be search. During the
search process, the DSBox instantiates multiple concrete D3M pipelines from DSBox
templates.

Here is an example of a simple template with each step bound to one primitive:

```python
{
	"name": "a_sample_template",  # Name of the template
	"taskType": "classification",  # Task type the template can handle
	"taskSubtype: {"binary", "multiClass"},  # Task subtypes the template can handle
	"inputType": "table",  # Input type the template can handle
    "output": "model_step",  # The name of the template step that generates the prediction
    "target": "extract_target_step",  # The name of the step that generates the ground truth
	"steps": [
        {
            "name": "denormalize_step",
            "primitives": ["d3m.primitives.dsbox.Denormalize"],
            "inputs": ["template_input"]
        },
        {
            "name": "to_dataframe_step",
            "primitives": ["d3m.primitives.datasets.DatasetToDataFrame"],
            "inputs": ["denormalize_step"]
        },
        {
            "name": "target",
            "primitives": [{
                "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                "hyperparameters":
                    {
                        'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',),
                        'use_columns': (),
                        'exclude_columns': ()
                    }
            }],
            "inputs": ["to_dataframe_step"]
        },
        {
            "name": "extract_attribute_step",
            "primitives": [{
                "primitive": "d3m.primitives.data.ExtractColumnsBySemanticTypes",
                "hyperparameters":
                    {
                        'semantic_types': (
                            'https://metadata.datadrivendiscovery.org/types/Attribute',),
                        'use_columns': (),
                        'exclude_columns': ()
                    }
            }],
            "inputs": ["to_dataframe_step"]
        },
        {
            "name": "model_step",
			"prmitives": [{
				"primitive": "d3m.primitives.sklearn_wrap.SKRandomForestClassifier",
				"hyperparameters":
					{
						'bootstrap': [True, False],
						'max_depth': [15, 30, None],
						'min_samples_leaf': [1, 2, 4],
						'min_samples_split': [2, 5, 10],
						'max_features': ['auto', 'sqrt'],
						'n_estimators': [10, 50, 100],
					}
			}],
			"inputs": ["extract_attribute_step", "target"]
		}
	]
}
```

The template snippet of the `model_step` below is an example of a DSBox template step bound to multiple primitives. This step can instantiated with three posible primitives:
`d3m.primitives.sklearn_wrap.SKRandomForestClassifier`,
`d3m.primitives.sklearn_wrap.SKExtraTreesClassifier` and
`d3m.primitives.sklearn_wrap.SKGradientBoostingClassifier`. The hyperparameter
dictionaries associated with each primitive define the space of possible
hyperparameters. And, this step takes two inputs, where are the outputs of the `data` and
`target` steps.

```javascript
{
    "name": "model_step",
    "primitives": [
        {
            "primitive":
            "d3m.primitives.sklearn_wrap.SKRandomForestClassifier",
            "hyperparameters":
            {
                'bootstrap': [True, False],
                'max_depth': [15, 30, None],
                'min_samples_leaf': [1, 2, 4],
                'min_samples_split': [2, 5, 10],
                'max_features': ['auto', 'sqrt'],
                'n_estimators': [10, 50, 100],
            }
        },
        {
            "primitive":
            "d3m.primitives.sklearn_wrap.SKExtraTreesClassifier",
            "hyperparameters":
            {
                'bootstrap': [True, False],
                'max_depth': [15, 30, None],
                'min_samples_leaf': [1, 2, 4],
                'min_samples_split': [2, 5, 10],
                'max_features': ['auto', 'sqrt'],
                'n_estimators': [10, 50, 100],
            }
        },
        {
            "primitive":
            "d3m.primitives.sklearn_wrap.SKGradientBoostingClassifier",
            "hyperparameters":
            {
                'max_depth': [2, 3, 4, 5],
                'n_estimators': [50, 60, 80, 100],
                'learning_rate': [0.1, 0.2, 0.4, 0.5],
                'min_samples_split': [2, 3],
                'min_samples_leaf': [1, 2],
            }
        },
    ],
    "inputs": ["data", "target"]
}
```
