#!/bin/bash
testType="size"
sizes=( 5000 )


for size in ${sizes[*]} 
do
  echo ${size}
  python sizeTime.py --size ${size}
done
