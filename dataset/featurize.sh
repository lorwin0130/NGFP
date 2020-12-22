#!/bin/bash
# get all filename in specified path

path=$1
files=$(ls $path)
for filename in $files
do
 ff_filename=${filename%\.*}.ff
#  echo $ff_filename >> filename.txt
 featurize pdb/$filename > ff/$ff_filename
done