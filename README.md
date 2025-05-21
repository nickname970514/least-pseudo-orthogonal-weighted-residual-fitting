# least-pseudo-orthogonal-weighted-residual-fitting

Fitting the variables x and y while taking into account their upper and lower errors, without averaging the upper and lower errors. This is necessary when there is a significant difference between the upper and lower errors.
The code also provides error scatter and intrinsic scatter of the data. Details can be found in the paper: (will be updated soon)

The code prioritizes the need to reproduce the content of the paper, and by default, it employs linear fitting. If you wish to skip the bootstrap process, you can simply set the last input value of the main function to None. To facilitate adjustments to the plotting effects, you can input the previously calculated values into the second-to-last, third-to-last, fourth-to-last, and fifth-to-last input values of the main function. This allows you to skip the fitting process while still retaining the information in the plot.

Additional Requirement:  

When using this code in academic work, you must cite the following paper:  (will be updated soon)
