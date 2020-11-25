# LLORMA-Trail
Simulation test setting code example for paper:

The simulation testing data is generated from \LLORMA-Trail\data\ptmatrix-500\L1\test\fitMain.py, settings are already written in the bash files.

The testing pipeline need .csv data file to use as input.

The example of \LLORMA-Trail\sizeTime.py shows how we transformed simulated data into .dat file that LLORMA method demand, and how to generate ground truth for simulation data.

You may want to copy these code to your own script if you want to run the data.

To run completely new data(different settings/name, etc.), you will also have to add lines of KIND and INIT as \LLORMA-Trail\base\dataset.py listed, one by one.
FYI pay attention to the form of names.

The KIND&INIT of testing data mentioned in the paper is already added to \LLORMA-Trail\base\dataset.py

The evaluation code is given in this GitHub repository in different settings.(xxxxEval.py)

The evaluation is based on the p and q vector that the LLORMA generate. Thus the test run commented some of the futher steps of LLORMA after p and q are made.




Original LLORMA method code from: https://github.com/JoonyoungYi/LLORMA-tensorflow

