#!/bin/bash

rsync -vaP -e "ssh" matrix.ml.cmu.edu:~/ranseq/100_*.txt .
