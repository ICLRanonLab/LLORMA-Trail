#!/bin/bash
testType="size"

times=( 0 1 )
errors=( 1 5 )


for errorStdBias in ${errors[*]} 
do
  for baseTimes in ${times[*]}
  do 
    for ((xn=10; xn<=50; xn+=5)); 
    do
      echo ${xn} ${baseTimes} ${errorStdBias} ${testType}
      python fitMain.py --xn ${xn} --baseTimes ${baseTimes} --errorStdBias ${errorStdBias} --testType ${testType}
    done
  done
done
