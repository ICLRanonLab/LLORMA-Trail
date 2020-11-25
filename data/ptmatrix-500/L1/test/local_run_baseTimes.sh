#!/bin/bash
#CALL conda.bat activate py36tf14
testType="baseTimes"

sizes=( 200 500 )
errors=( 0 1 )


for xn in ${sizes[*]} 
do
  for errorStdBias in ${errors[*]} 
  do  
    for ((baseTimes=0; baseTimes<=30; baseTimes+=1)); 
    do
      echo ${xn} ${baseTimes} ${errorStdBias} ${testType}
      python fitMain.py --xn ${xn} --baseTimes ${baseTimes} --errorStdBias ${errorStdBias} --testType ${testType}
    done
  done
done
