#!/bin/bash
testType="err"
means=( 0 1 )
sizes=(20 50)
# errors=( 0 1 2 3 4 5 6 7 8 9 10 15 20 30 40 )
#for ((xn=500; xn<=500; xn+=50)); do
for xn in ${sizes[*]} 
do
  #for ((baseTimes = 1; baseTimes <= 1; baseTimes+=1)); do
  for baseTimes in ${means[*]} 
  do 
    for ((error=0; error<=30; error+=1));
    do
      echo ${xn} ${baseTimes} ${errorStdBias} ${testType}
      python fitMain.py --xn ${xn} --baseTimes ${baseTimes} --errorStdBias ${error} --testType ${testType}
    done
  done
done
