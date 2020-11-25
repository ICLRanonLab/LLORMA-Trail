#!/bin/bash
#CALL conda.bat activate py36tf14
testType="basetimes"

sizes=( 20 50 )
errors=( 0 1 )


for size in ${sizes[*]} 
do
  for error in ${errors[*]} 
  do  
    for ((baseTimes=0; baseTimes<=30; baseTimes+=1)); 
    do
      echo ${size} ${baseTimes} ${error} ${testType}
      python meanSizeSquareEval.py --size ${size} --mean ${baseTimes} --error ${error} --testType ${testType}
    done
  done
done
