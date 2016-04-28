Running instructions:
For each dataset, run the names jupyter notebook. You can run clean.sh at the 
end to remove all the junk that is populated by my scripts.

Also, set environment CAFFE_ROOT to point to location of Caffe. I am also
shipping a custom version of the convert_imageset exe that is parallelized and
handles filenames with spaces. For this tool to run, the location of caffe's
shared library file needs to be added to the LD_LIBRARY_PATH. These two
variables can also be set in run_training_pipeline.sh to point to the correct
location on your setup.
