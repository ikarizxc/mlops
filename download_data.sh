#!/bin/bash

kaggle competitions download -c home-credit-default-risk
mkdir data
mv home-credit-default-risk.zip data
unzip data/home-credit-default-risk.zip -d data