# pytorch-rec
Matrix factorization models in pytorch

Install steps

1. You'll need to install [pytorch](http://pytorch.org/) seperately
2. pip3 -r requirements.txt
3. cd data; python3 download.py  # This will take 1-2Gb of disk space
4. python3 run.py

Included factorization models:
  1. MF. A simple matrix factorization model with user and item biases + vectors
  1. FM. A simple factorization machine model with user, item, and genre interactions
  2. MFPoly2. Adds a 2nd order polynomial on age.
  3. MFPoincare. Matrix factorization in Poincare (instead of Euclidean) space
  4. SparseMF.

Included graph autoencoder models:
    1. SimpleAE
    2. LSTMAE
    3. ActiveAE
