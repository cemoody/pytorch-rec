# pytorch-rec
Matrix factorization models in pytorch

Install steps

1. You'll need to install [pytorch](http://pytorch.org/) seperately
2. pip3 -r requirements.txt
3. python3 download.py  # This will take 1-2Gb of disk space
4. python3 run.py

Included models:
  1. MF. A simple matrix factorization model with user and item biases + vectors
  2. MFPoly2. Adds a 2nd order polynomial on age.
  3. MFPoincare. Matrix factorization in Poincare (instead of Euclidean) space
