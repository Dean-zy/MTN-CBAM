# MTN-CBAM
A multi-task net for KWS and speaker detection.
The description of this experiment is in the paper：
“MTN-CBAM: Multi-Task Network with Convolutional Block Attention Module for Speaker Related Small-Footprint Keyword Spotting”
# Environment configuration
In order to run these Python scripts, the following libraries and packages are needed:
	* Keras
	* Librosa
	* Numpy
	* Pickle
	* Matplotlib
# Data and directory  
When running these Python scripts, by default, it is expected to find two
	folders within this one: "HADataset" and "exp". The first would contain
	the hearing aid speech database that can be freely downloaded from [Data Link](https://drive.google.com/file/d/1EMgPGZZm1TKtYqSxVnTQ90DsZTjjmPmx/view?usp=sharing)
	The second folder is the working directory, where all files resulting from running the
	provided scripts are stored.

Thanks to [Lopezespejo I, Tan Z, Jensen J, et al. Keyword Spotting for Hearing Assistive Devices Robust to External Speakers[C]. conference of the international speech communication association, 2019: 3223-3227](https://cn.bing.com/academic/profile?id=5897b33944ec0496f93494c64a246f9d&encoded=0&v=paper_preview&mkt=zh-cn).  for providing this data set

# How to run
run.sh demonstrates the running example
