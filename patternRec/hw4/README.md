Running instructions:
Requirements:
Python2 + usual data analysis libraries
Caffe + PyCaffe

For each experiment, run the correspondingly named jupyter notebook
(mnist1, mnist2, sunset, office). You can run clean.sh at the end to remove all
the junk that is populated by my scripts.

Here is my layout for the data folder that should be mimicked (including case):
<ROOT>
  |-> data
    |-> mnist
    |-> Office4D2
    |-> Sunset

Also, set environment CAFFE_ROOT to point to location of Caffe. I am also
shipping a custom version of the convert_imageset exe that is parallelized and
handles filenames with spaces. For this tool to run, the location of caffe's
shared library file needs to be added to the LD_LIBRARY_PATH. These two
variables can also be set in run_training_pipeline.sh to point to the correct
location on your setup.
