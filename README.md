# EigenThermal
PCA-based experiment on a dataset of thermal images.

The implementation computes:
* the average route
* the eigenImages corresponding to the 10 largest eigenvalues
* the eigenImages corresponding to the 10 smallest eigenvalues

### Requirements
OpenCV library

Compiles with: 
g++ eigenthermal.cpp `pkg-config --cflags --libs opencv`
