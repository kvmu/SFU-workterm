# NSERC USRA Workterm: Looking at the Higgs -> WW* Boson decay channel through vector boson fusion (VBF)
Here I have a collection of my work conducted at SFU during the summer of 2015. To get some context and a good summary of my work, download the final_poster.pdf and have a look! It summarizes everything neatly (if you try to read it through GitHub it looks weird due to the some rendering issue).

Below, I just have some description of the functions I have put up.

## factory_analysis.py

factory_analysis.py contains the main code that implements
the matrix element method. The file contains parameters that
control the integration, paths for data output, number of events,
what cuts to apply on the data, etc.

Currently RUN I (CERN Experiment run 1) ntuples (data structure) are used to do the analysis, as opposed
to mg5 (MadGraph -- it is a software that generates particle interaction events using the Feynmann diagrams) generated events.

Right now, VEGAS creates a phase space (line 199) and then
for each event, that same space is used. Per event, the calculation
is run until a condition is reached (line 212), otherwise each event
runs 5 times. For the condition to be met, there needs to be a
sufficiently large 'neval'. The adapted phase space is used for
each consecutive event.

For further details, the file contains many comments.

## my_timer.py

A simple timer context manager, used for rough profiling of code.

## GPU_compare_CPU.py

This script does CPU and GPU matrix element time complexity
profiling. It has a function which applies the matrix element
analysis for a given set of parameters, profiles the code and
plots the time complexity results (with fit) and plots the matrix
elements from each case.

## gen_plots.py

This script takes the data outputted by factory_analysis.py
(from the directory vbf_plotting) and then creates the histograms
of the discriminant given by the Neymann-Pearson Lemma
 (log(ME_signal/ME_bkg)), outputting the plot in the vbf_plotting/
 directory. This script also creates the ROC curve and gives a
 value for the separation which is bin-independent (only depends
 on the ROC curve).

 More details can be found in the file.

## LDA_compare_Ratio.py

## kernel.py
