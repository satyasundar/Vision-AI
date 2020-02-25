Group Members: Satya Nayak, Ramjee Ganti, Gourav Pattanaik, Jayant Ojha

Please refer the [notebook](https://github.com/ojhajayant/eva/blob/master/week6/S6_assignment.ipynb) for this assignment solution.

### Finding optimum values for L1-penalty-weight & L2-weight-decay
    - The appropriate values for l1 & l2 related weights were found after coarsely sweeping
      thru values from 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001 first.
    - For l1-weight, with l1-alone enabled, the 0.00001 region was found with better values
      for the validation accuracy vis-a-vis training accuracies.
    - Much finer sweeping in the 0.00001 region, led to the value: 0.000025, which seems to
      be having best accuracy values.
    - Similarly, For l2-weight,  with l2-alone enabled, the 0.0001 region was found with better
      values for the validation accuracy vis-a-vis training accuracies.
    - Much finer sweeping in the 0.0001 region, led to the value: 0.0002125, which seems to
      be having best accuracy values.
    - For individual models, either or both of them will be enabled/disbaled using some
      related flags defined ahead in this notebook
    - Please note, L2 regaularization implementation is in-built with torch. 
The experiment results for above sweeps are  [here](https://github.com/ojhajayant/eva/blob/master/week6/S5_assignment_3rd_attempt_40EPOCH_l1_only_experiments.ipynb) & [here](https://github.com/ojhajayant/eva/blob/master/week6/S5_assignment_3rd_attempt_40EPOCH_l2_only_experiments.ipynb)

###  2 graphs to show the validation accuracy change and loss change:
![alt text](https://github.com/ojhajayant/eva/blob/master/week6/test_accuracies.PNG "Logo Title Text 1")

![alt text](https://github.com/ojhajayant/eva/blob/master/week6/test_losses.PNG "Logo Title Text 1")

###  misclassified images:
![alt text](https://github.com/ojhajayant/eva/blob/master/week6/Predicted%20_Vs_%20Actual_Without_both_%20L1_n_L2.PNG "Logo Title Text 1")
![alt text](https://github.com/ojhajayant/eva/blob/master/week6/Predicted%20_Vs_%20Actual_With_L1_alone.PNG "Logo Title Text 1")
![alt text](https://github.com/ojhajayant/eva/blob/master/week6/Predicted%20_Vs_%20Actual_With_L2_alone.PNG "Logo Title Text 1")
![alt text](https://github.com/ojhajayant/eva/blob/master/week6/Predicted%20_Vs_%20Actual_With_both_%20L1_n_L2.PNG "Logo Title Text 1")


