#!/bin/bash
testType="size"
means=( 0 1 )
errors=( 1 5 )

for error in ${errors[*]} 
do
  #for ((baseTimes = 1; baseTimes <= 1; baseTimes+=1)); do
  for mean in ${means[*]} 
  do 
    for ((size=10; size<=50; size+=5));
    do
      echo ${size} ${baseTimes} ${error} ${testType}
      python sizeSizeSquareEval.py --size ${size} --mean ${mean} --error ${error} --testType ${testType}
    done
  done
done