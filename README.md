## New Patient Models

Code: `new_patient_model.py`, `new_patient_additional_model_descs.py`, `simplified_models.py`

Data: `patient_data_venex/` (currently private)


## Analysis of patient data

Data: `patient_data_venex/` (currently private)

Notebooks: `publication_notebooks/`


## Helper code

`tellurium_model_fitting.py` - this contains code for constructing objective functions.

`find_map.py` - this is a hack of PyMC's `find_MAP` function that allows for additional gradient-free optimization methods, such as PyBOBY-QA.

`systematic_model_comparisons_multiprocessing.py` - this is a script for running the models on the dataset.
