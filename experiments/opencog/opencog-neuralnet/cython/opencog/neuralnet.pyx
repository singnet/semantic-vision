# To make single library from few .pyx modules include them in one .pyx module
# which name is equal to library name. All exported C/C++ definitions should be
# placed into corresponding .pxd file. Name of the .so file should have same
# name as this .pyx module.

include "ptrvalue.pyx"
