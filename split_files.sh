#!/bin/bash

FN=train_covariates.tsv
split -l 500 -d $FN covars_all/part


FN=train_observed_labels.tsv
split -l 500 -d $FN labels_all/part


FN=train_experiment_ids.tsv
split -l 500 -d $FN ids_all/part
