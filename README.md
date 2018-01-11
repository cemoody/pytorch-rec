# pytorch-rec
Matrix factorization models in pytorch

Install steps

1. You'll need to install [pytorch](http://pytorch.org/) seperately
2. virtualenv --python=python3.6 env
2. source env/bin/activate
2. pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl  # Install as appropriate
3. pip3 install torchvision
2. pip3 -r requirements.txt
3. cd data; python3 download.py  # This will take 1-2Gb of disk space
4. python3 run.py

Included factorization models:
  1. MF. A simple matrix factorization model with user and item biases + vectors
  3. FM. A simple factorization machine model with user, item, and genre interactions
  2. VFM.  + variational, lots of structure & hyperpriors
  4. MFPoly2. Adds a 2nd order polynomial on age.
  5. GumbleMF. Emphasizes interpretable topics and vectors
  6. MFPoincare. Matrix factorization in Poincare (instead of Euclidean) space

Included graph autoencoder models:
    1. SimpleAE
    2. LSTMAE
    3. ActiveAE

Extras:
    Poisson likelihood
    Multi-output models
