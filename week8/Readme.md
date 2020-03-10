Group Members: Satya Nayak, Ramjee Ganti, Gourav Pattanaik, Jayant Ojha

    For the week7 assignment, I had submitted it in a python-script-mode, rather than the required
    ipynb-notebook format [after the package installation (hosted @ github)], hence had to make
    small modifications in the week7 submission itself, to make it installable (had to add setup.py,
    few empty __init__.py files & some changes in the way & order, in which local imports were made)
    Later on, to accomodate the week8-assignment's requirement, to have the ResNet18-NW incorporated,
    from the given github repo: https://github.com/kuangliu/pytorch-cifar, have shifted this 
    additional stuff under the "week8"-package, with all the rest, leveraged from week7. Hence, below,
    am making imports from "week8.modular". The submitted notebook file, represents the "main.py",
    which was the main script to be run in my last submission.

Please refer [this notebook](https://github.com/ojhajayant/eva/blob/master/week8/S8_assignment.ipynb) for the code.

The ResNet18 model has been taken from [this link](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py).
But as I had been using the nn.NLLLoss() criterion (rather than torch.nn.CrossEntropyLoss) hence additional change of
returning the nn.LogSoftmax() for the given ResNet18 'out', had to be done, as an only change.
Only the following img-augmentation methods are used: transforms.RandomCrop(32, padding=4) & transforms.RandomHorizontalFlip()
Haven't yet started using the main-cutout method for the same.
```
    final accuracy of your model = 89.61%
    Here are the training logs:
      
      	0%|                                                                                          | 0/782 [00:00<?, ?it/s]
	Model training starts on CIFAR10 dataset
	EPOCH: 1
	Loss=1.2107690572738647 Batch_id=781 Accuracy=41.43: 100%|███████████████████████████| 782/782 [01:56<00:00,  6.72it/s]

	Test set: Average loss: 1.2744, Accuracy: 5338/10000 (53.38%)

	validation-accuracy improved from 0 to 53.38, saving model to D:\PG-ML\eva4\week8\./saved_models/CIFAR10_model_epoch-1_L1-1_L2-0_val_acc-53.38.h5
	EPOCH: 2
	Loss=1.4697918891906738 Batch_id=781 Accuracy=59.03: 100%|███████████████████████████| 782/782 [01:57<00:00,  6.64it/s]

	Test set: Average loss: 1.0451, Accuracy: 6288/10000 (62.88%)

	validation-accuracy improved from 53.38 to 62.88, saving model to D:\PG-ML\eva4\week8\./saved_models/CIFAR10_model_epoch-2_L1-1_L2-0_val_acc-62.88.h5
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]
	EPOCH: 3
	Loss=0.9282803535461426 Batch_id=781 Accuracy=66.83: 100%|███████████████████████████| 782/782 [01:58<00:00,  6.62it/s]

	Test set: Average loss: 0.9193, Accuracy: 6775/10000 (67.75%)

	validation-accuracy improved from 62.88 to 67.75, saving model to D:\PG-ML\eva4\week8\./saved_models/CIFAR10_model_epoch-3_L1-1_L2-0_val_acc-67.75.h5
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]
	EPOCH: 4
	Loss=0.8228573203086853 Batch_id=781 Accuracy=72.04: 100%|███████████████████████████| 782/782 [01:58<00:00,  6.60it/s]

	Test set: Average loss: 0.8337, Accuracy: 7177/10000 (71.77%)

	validation-accuracy improved from 67.75 to 71.77, saving model to D:\PG-ML\eva4\week8\./saved_models/CIFAR10_model_epoch-4_L1-1_L2-0_val_acc-71.77.h5
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]
	EPOCH: 5
	Loss=1.0518923997879028 Batch_id=781 Accuracy=75.75: 100%|███████████████████████████| 782/782 [01:59<00:00,  6.56it/s]
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

	Test set: Average loss: 0.8460, Accuracy: 7162/10000 (71.62%)

	EPOCH: 6
	Loss=0.7382607460021973 Batch_id=781 Accuracy=78.18: 100%|███████████████████████████| 782/782 [02:00<00:00,  6.52it/s]

	Test set: Average loss: 0.6492, Accuracy: 7762/10000 (77.62%)

	validation-accuracy improved from 71.77 to 77.62, saving model to D:\PG-ML\eva4\week8\./saved_models/CIFAR10_model_epoch-6_L1-1_L2-0_val_acc-77.62.h5
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]
	EPOCH: 7
	Loss=0.4634123742580414 Batch_id=781 Accuracy=80.03: 100%|███████████████████████████| 782/782 [01:59<00:00,  6.53it/s]

	Test set: Average loss: 0.5757, Accuracy: 8057/10000 (80.57%)

	validation-accuracy improved from 77.62 to 80.57, saving model to D:\PG-ML\eva4\week8\./saved_models/CIFAR10_model_epoch-7_L1-1_L2-0_val_acc-80.57.h5
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]
	EPOCH: 8
	Loss=0.877304196357727 Batch_id=781 Accuracy=81.69: 100%|████████████████████████████| 782/782 [01:59<00:00,  6.54it/s]

	Test set: Average loss: 0.5643, Accuracy: 8111/10000 (81.11%)

	validation-accuracy improved from 80.57 to 81.11, saving model to D:\PG-ML\eva4\week8\./saved_models/CIFAR10_model_epoch-8_L1-1_L2-0_val_acc-81.11.h5
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]
	EPOCH: 9
	Loss=0.7454615831375122 Batch_id=781 Accuracy=83.04: 100%|███████████████████████████| 782/782 [01:59<00:00,  6.53it/s]
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

	Test set: Average loss: 0.5655, Accuracy: 8097/10000 (80.97%)

	EPOCH: 10
	Loss=0.32246485352516174 Batch_id=781 Accuracy=84.10: 100%|██████████████████████████| 782/782 [02:00<00:00,  6.48it/s]

	Test set: Average loss: 0.5338, Accuracy: 8247/10000 (82.47%)

	validation-accuracy improved from 81.11 to 82.47, saving model to D:\PG-ML\eva4\week8\./saved_models/CIFAR10_model_epoch-10_L1-1_L2-0_val_acc-82.47.h5
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]
	EPOCH: 11
	Loss=0.34272873401641846 Batch_id=781 Accuracy=85.17: 100%|██████████████████████████| 782/782 [02:01<00:00,  6.45it/s]

	Test set: Average loss: 0.4735, Accuracy: 8435/10000 (84.35%)

	validation-accuracy improved from 82.47 to 84.35, saving model to D:\PG-ML\eva4\week8\./saved_models/CIFAR10_model_epoch-11_L1-1_L2-0_val_acc-84.35.h5
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]
	EPOCH: 12
	Loss=0.6178702116012573 Batch_id=781 Accuracy=86.08: 100%|███████████████████████████| 782/782 [02:00<00:00,  6.49it/s]
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

	Test set: Average loss: 0.4826, Accuracy: 8389/10000 (83.89%)

	EPOCH: 13
	Loss=0.40690183639526367 Batch_id=781 Accuracy=86.97: 100%|██████████████████████████| 782/782 [02:00<00:00,  6.47it/s]

	Test set: Average loss: 0.4471, Accuracy: 8496/10000 (84.96%)

	validation-accuracy improved from 84.35 to 84.96, saving model to D:\PG-ML\eva4\week8\./saved_models/CIFAR10_model_epoch-13_L1-1_L2-0_val_acc-84.96.h5
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]
	EPOCH: 14
	Loss=0.3110704720020294 Batch_id=781 Accuracy=87.59: 100%|███████████████████████████| 782/782 [02:00<00:00,  6.48it/s]

	Test set: Average loss: 0.4623, Accuracy: 8501/10000 (85.01%)

	validation-accuracy improved from 84.96 to 85.01, saving model to D:\PG-ML\eva4\week8\./saved_models/CIFAR10_model_epoch-14_L1-1_L2-0_val_acc-85.01.h5
	EPOCH: 15
	Loss=0.2781562805175781 Batch_id=781 Accuracy=88.09: 100%|███████████████████████████| 782/782 [02:00<00:00,  6.48it/s]
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

	Test set: Average loss: 0.4730, Accuracy: 8459/10000 (84.59%)

	EPOCH: 16
	Loss=0.21431131660938263 Batch_id=781 Accuracy=88.86: 100%|██████████████████████████| 782/782 [02:01<00:00,  6.44it/s]
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

	Test set: Average loss: 0.4704, Accuracy: 8457/10000 (84.57%)

	EPOCH: 17
	Loss=0.3530069589614868 Batch_id=781 Accuracy=89.25: 100%|███████████████████████████| 782/782 [02:01<00:00,  6.46it/s]

	Test set: Average loss: 0.4224, Accuracy: 8635/10000 (86.35%)

	validation-accuracy improved from 85.01 to 86.35, saving model to D:\PG-ML\eva4\week8\./saved_models/CIFAR10_model_epoch-17_L1-1_L2-0_val_acc-86.35.h5
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]
	EPOCH: 18
	Loss=0.2505944073200226 Batch_id=781 Accuracy=90.02: 100%|███████████████████████████| 782/782 [02:00<00:00,  6.48it/s]
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

	Test set: Average loss: 0.4629, Accuracy: 8561/10000 (85.61%)

	EPOCH: 19
	Loss=0.27494993805885315 Batch_id=781 Accuracy=90.41: 100%|██████████████████████████| 782/782 [02:00<00:00,  6.48it/s]
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

	Test set: Average loss: 0.4472, Accuracy: 8568/10000 (85.68%)

	EPOCH: 20
	Loss=0.18553119897842407 Batch_id=781 Accuracy=90.89: 100%|██████████████████████████| 782/782 [02:00<00:00,  6.47it/s]

	Test set: Average loss: 0.4097, Accuracy: 8705/10000 (87.05%)

	validation-accuracy improved from 86.35 to 87.05, saving model to D:\PG-ML\eva4\week8\./saved_models/CIFAR10_model_epoch-20_L1-1_L2-0_val_acc-87.05.h5
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]
	EPOCH: 21
	Loss=0.8716439008712769 Batch_id=781 Accuracy=91.14: 100%|███████████████████████████| 782/782 [02:00<00:00,  6.49it/s]

	Test set: Average loss: 0.3959, Accuracy: 8745/10000 (87.45%)

	validation-accuracy improved from 87.05 to 87.45, saving model to D:\PG-ML\eva4\week8\./saved_models/CIFAR10_model_epoch-21_L1-1_L2-0_val_acc-87.45.h5
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]
	EPOCH: 22
	Loss=0.5520456433296204 Batch_id=781 Accuracy=91.54: 100%|███████████████████████████| 782/782 [02:00<00:00,  6.47it/s]
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

	Test set: Average loss: 0.4014, Accuracy: 8719/10000 (87.19%)

	EPOCH: 23
	Loss=0.5016188621520996 Batch_id=781 Accuracy=92.11: 100%|███████████████████████████| 782/782 [02:00<00:00,  6.47it/s]

	Test set: Average loss: 0.3924, Accuracy: 8754/10000 (87.54%)

	validation-accuracy improved from 87.45 to 87.54, saving model to D:\PG-ML\eva4\week8\./saved_models/CIFAR10_model_epoch-23_L1-1_L2-0_val_acc-87.54.h5
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]
	EPOCH: 24
	Loss=0.42992937564849854 Batch_id=781 Accuracy=92.42: 100%|██████████████████████████| 782/782 [02:01<00:00,  6.44it/s]
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

	Test set: Average loss: 0.4069, Accuracy: 8715/10000 (87.15%)

	EPOCH: 25
	Loss=0.18324445188045502 Batch_id=781 Accuracy=92.65: 100%|██████████████████████████| 782/782 [02:00<00:00,  6.47it/s]
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

	Test set: Average loss: 0.3910, Accuracy: 8746/10000 (87.46%)

	EPOCH: 26
	Loss=0.09726032614707947 Batch_id=781 Accuracy=93.04: 100%|██████████████████████████| 782/782 [02:01<00:00,  6.42it/s]

	Test set: Average loss: 0.3848, Accuracy: 8819/10000 (88.19%)

	validation-accuracy improved from 87.54 to 88.19, saving model to D:\PG-ML\eva4\week8\./saved_models/CIFAR10_model_epoch-26_L1-1_L2-0_val_acc-88.19.h5
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]
	EPOCH: 27
	Loss=0.6072551608085632 Batch_id=781 Accuracy=93.20: 100%|███████████████████████████| 782/782 [02:02<00:00,  6.40it/s]
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

	Test set: Average loss: 0.4156, Accuracy: 8763/10000 (87.63%)

	EPOCH: 28
	Loss=0.08645926415920258 Batch_id=781 Accuracy=93.58: 100%|██████████████████████████| 782/782 [02:02<00:00,  6.40it/s]
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

	Test set: Average loss: 0.3892, Accuracy: 8817/10000 (88.17%)

	EPOCH: 29
	Loss=0.5381070971488953 Batch_id=781 Accuracy=94.04: 100%|███████████████████████████| 782/782 [02:02<00:00,  6.37it/s]

	Test set: Average loss: 0.3838, Accuracy: 8838/10000 (88.38%)

	validation-accuracy improved from 88.19 to 88.38, saving model to D:\PG-ML\eva4\week8\./saved_models/CIFAR10_model_epoch-29_L1-1_L2-0_val_acc-88.38.h5
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]
	EPOCH: 30
	Loss=0.4520813524723053 Batch_id=781 Accuracy=94.12: 100%|███████████████████████████| 782/782 [02:04<00:00,  6.27it/s]
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

	Test set: Average loss: 0.4254, Accuracy: 8728/10000 (87.28%)

	EPOCH: 31
	Loss=0.09137478470802307 Batch_id=781 Accuracy=94.44: 100%|██████████████████████████| 782/782 [02:04<00:00,  6.29it/s]
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

	Test set: Average loss: 0.4382, Accuracy: 8699/10000 (86.99%)

	EPOCH: 32
	Loss=0.14652249217033386 Batch_id=781 Accuracy=94.67: 100%|██████████████████████████| 782/782 [02:02<00:00,  6.39it/s]

	Test set: Average loss: 0.3850, Accuracy: 8864/10000 (88.64%)

	validation-accuracy improved from 88.38 to 88.64, saving model to D:\PG-ML\eva4\week8\./saved_models/CIFAR10_model_epoch-32_L1-1_L2-0_val_acc-88.64.h5
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]
	EPOCH: 33
	Loss=0.052406638860702515 Batch_id=781 Accuracy=94.93: 100%|█████████████████████████| 782/782 [02:02<00:00,  6.39it/s]
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

	Test set: Average loss: 0.4366, Accuracy: 8756/10000 (87.56%)

	EPOCH: 34
	Loss=0.06312966346740723 Batch_id=781 Accuracy=95.45: 100%|██████████████████████████| 782/782 [02:02<00:00,  6.41it/s]

	Test set: Average loss: 0.3845, Accuracy: 8886/10000 (88.86%)

	validation-accuracy improved from 88.64 to 88.86, saving model to D:\PG-ML\eva4\week8\./saved_models/CIFAR10_model_epoch-34_L1-1_L2-0_val_acc-88.86.h5
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]
	EPOCH: 35
	Loss=0.019796371459960938 Batch_id=781 Accuracy=95.24: 100%|█████████████████████████| 782/782 [02:02<00:00,  6.40it/s]
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

	Test set: Average loss: 0.3957, Accuracy: 8873/10000 (88.73%)

	EPOCH: 36
	Loss=0.021409928798675537 Batch_id=781 Accuracy=95.50: 100%|█████████████████████████| 782/782 [02:01<00:00,  6.42it/s]
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

	Test set: Average loss: 0.4048, Accuracy: 8830/10000 (88.30%)

	EPOCH: 37
	Loss=0.18261584639549255 Batch_id=781 Accuracy=95.78: 100%|██████████████████████████| 782/782 [02:02<00:00,  6.40it/s]

	Test set: Average loss: 0.3791, Accuracy: 8941/10000 (89.41%)

	validation-accuracy improved from 88.86 to 89.41, saving model to D:\PG-ML\eva4\week8\./saved_models/CIFAR10_model_epoch-37_L1-1_L2-0_val_acc-89.41.h5
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]
	EPOCH: 38
	Loss=0.07097992300987244 Batch_id=781 Accuracy=95.86: 100%|██████████████████████████| 782/782 [02:01<00:00,  6.42it/s]
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]

	Test set: Average loss: 0.4104, Accuracy: 8871/10000 (88.71%)

	EPOCH: 39
	Loss=0.2160685807466507 Batch_id=781 Accuracy=96.12: 100%|███████████████████████████| 782/782 [02:01<00:00,  6.45it/s]

	Test set: Average loss: 0.3783, Accuracy: 8961/10000 (89.61%)

	validation-accuracy improved from 89.41 to 89.61, saving model to D:\PG-ML\eva4\week8\./saved_models/CIFAR10_model_epoch-39_L1-1_L2-0_val_acc-89.61.h5
	  0%|                                                                                          | 0/782 [00:00<?, ?it/s]
	EPOCH: 40
	Loss=0.05721578001976013 Batch_id=781 Accuracy=96.23: 100%|██████████████████████████| 782/782 [02:01<00:00,  6.46it/s]

	Test set: Average loss: 0.3810, Accuracy: 8946/10000 (89.46%)

```
Here are the train/test-loss and Train/test-accuracy plots over a 40-epoch-run:

![alt text](https://github.com/ojhajayant/eva/blob/master/week8/acc_loss.PNG "Logo Title Text 1")

Here are example confusion-matrix and classification-report:

![alt text](https://github.com/ojhajayant/eva/blob/master/week8/classif_rpt.PNG "Logo Title Text 1")

Here are 3 groups of mislabelled 10-class samples from the test-dataset:

![alt text](https://github.com/ojhajayant/eva/blob/master/week8/mislabelled.PNG "Logo Title Text 1")
