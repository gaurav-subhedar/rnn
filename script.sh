#!/bin/bash
declare -i d

echo "Enter character password"
read userpwd
echo -e $userpwd | tee -a log.txt
size=${#userpwd}
echo "size:" $size
for (( c=0; c<size; c++ ))
do
b=${userpwd:0:c+1}
#echo -e $b
d=$size-$c
th sample.lua models/lm_lstm_epoch10.00_1.5924.t7 -gpuid -1 -prefix $b | tee -a log.txt
echo -e "" | tee -a log.txt
done
dirpath=$PWD
dirpath+="/log.txt"
echo $dirpath
java cns.calculateScore $dirpath 
