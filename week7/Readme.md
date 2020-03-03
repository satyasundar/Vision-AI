Group Members: Satya Nayak, Ramjee Ganti, Gourav Pattanaik, Jayant Ojha

Please refer [this location](https://github.com/ojhajayant/eva/tree/master/week7/modular) for the code.

Following shows the model, based on the C1-T1-C2-T2-C3-T3-C4-O template, and has a RF = 66x66 (greater than 44x44 as required) with ~58K parameters.

    - Each layer here uses Depth-wise Separable Convolution (hence the o/p capacity for each
      can be increased while keeping quite lesser parameters as compared to "tradional" 3x3 conv2d.
    - For reaching the required edges and gradient @ 7x7 (and for the overall NW to have a 
      RF > 44) the dilation for the 2nd layer is kept as 2.
    - Rest all of the layers have a dilation of 1.
    - The C4-block is a "capacity-booster" element, providing 64 feature points for the 
      fully-connected layer & log-softmax to classify across 10 classes.

![alt text](https://github.com/ojhajayant/eva/blob/master/week7/model_dgm.png "Logo Title Text 1")

The directory-structure for the code is as  below:

![alt text](https://github.com/ojhajayant/eva/blob/master/week7/dir_struct.png "Logo Title Text 1")
```
0. cfg.py: This has the default and/or user-supplied top-level configuration values & global-vars.
1. main.py: This is the main script to be run to either train or make inference.

            example usage is as below:
            python main.py train --SEED 2 --batch_size 64  --epochs 10 --lr 0.01 \
                                 --dropout 0.05 --l1_weight 0.00002  --l2_weight_decay 0.000125 \
                                 --L1 True --L2 False --data data --best_model_path saved_models \
                                 --prefix data

            python main.py test  --batch_size 64  --data data --best_model_path saved_models \
                                 --best_model 'CIFAR10_model_epoch-39_L1-1_L2-0_val_acc-81.83.h5' \
                                 --prefix data
2. network.py: This is the model-code.
3. preprocess.py: This code is to download & preprocess data.
4. utils.py: This has all the 'utility' code used across.
5. train.py: This has the model-training code.
6. test.py: This has the model-inference code.

The downloaded datset or any other intermediate plots or config.txt files are saved to the ./data (or 
user-provided folder)
The models are saved to the ./saved_models (or user-provided folder)
```
An example log from the train command is shown below:
```
EPOCH: 1
Loss=0.9716601967811584 Batch_id=781 Accuracy=46.20: 100%|███████████████████████████| 782/782 [00:39<00:00, 20.02it/s]

Test set: Average loss: 1.3070, Accuracy: 5345/10000 (53.45%)

validation-accuracy improved from 0 to 53.45, saving model to D:\PG-ML\eva4\week7\saved_models\CIFAR10_model_epoch-1_L1-1_L2-0_val_acc-53.45.h5
EPOCH: 2
Loss=1.3412349224090576 Batch_id=781 Accuracy=61.69: 100%|███████████████████████████| 782/782 [00:39<00:00, 20.00it/s]

Test set: Average loss: 1.1244, Accuracy: 6018/10000 (60.18%)

validation-accuracy improved from 53.45 to 60.18, saving model to D:\PG-ML\eva4\week7\saved_models\CIFAR10_model_epoch-2_L1-1_L2-0_val_acc-60.18.h5
EPOCH: 3
Loss=1.0065510272979736 Batch_id=781 Accuracy=67.21: 100%|███████████████████████████| 782/782 [00:39<00:00, 19.99it/s]

Test set: Average loss: 0.9458, Accuracy: 6657/10000 (66.57%)

validation-accuracy improved from 60.18 to 66.57, saving model to D:\PG-ML\eva4\week7\saved_models\CIFAR10_model_epoch-3_L1-1_L2-0_val_acc-66.57.h5
EPOCH: 4
Loss=0.4409160912036896 Batch_id=781 Accuracy=70.33: 100%|███████████████████████████| 782/782 [00:39<00:00, 19.78it/s]

Test set: Average loss: 0.8602, Accuracy: 7040/10000 (70.40%)

validation-accuracy improved from 66.57 to 70.4, saving model to D:\PG-ML\eva4\week7\saved_models\CIFAR10_model_epoch-4_L1-1_L2-0_val_acc-70.4.h5
EPOCH: 5
Loss=0.524391770362854 Batch_id=781 Accuracy=72.73: 100%|████████████████████████████| 782/782 [00:39<00:00, 19.95it/s]

Test set: Average loss: 0.7616, Accuracy: 7330/10000 (73.30%)

validation-accuracy improved from 70.4 to 73.3, saving model to D:\PG-ML\eva4\week7\saved_models\CIFAR10_model_epoch-5_L1-1_L2-0_val_acc-73.3.h5
EPOCH: 6
Loss=0.5638612508773804 Batch_id=781 Accuracy=74.69: 100%|███████████████████████████| 782/782 [00:39<00:00, 19.78it/s]

Test set: Average loss: 0.7749, Accuracy: 7274/10000 (72.74%)

EPOCH: 7
Loss=0.4855955243110657 Batch_id=781 Accuracy=76.15: 100%|███████████████████████████| 782/782 [00:39<00:00, 19.79it/s]

Test set: Average loss: 0.8081, Accuracy: 7194/10000 (71.94%)

EPOCH: 8
Loss=0.5581735968589783 Batch_id=781 Accuracy=77.43: 100%|███████████████████████████| 782/782 [00:39<00:00, 20.00it/s]

Test set: Average loss: 0.7667, Accuracy: 7334/10000 (73.34%)

validation-accuracy improved from 73.3 to 73.34, saving model to D:\PG-ML\eva4\week7\saved_models\CIFAR10_model_epoch-8_L1-1_L2-0_val_acc-73.34.h5
EPOCH: 9
Loss=0.5147180557250977 Batch_id=781 Accuracy=78.21: 100%|███████████████████████████| 782/782 [00:39<00:00, 19.99it/s]

Test set: Average loss: 0.6752, Accuracy: 7691/10000 (76.91%)

validation-accuracy improved from 73.34 to 76.91, saving model to D:\PG-ML\eva4\week7\saved_models\CIFAR10_model_epoch-9_L1-1_L2-0_val_acc-76.91.h5
EPOCH: 10
Loss=0.8015247583389282 Batch_id=781 Accuracy=79.06: 100%|███████████████████████████| 782/782 [00:39<00:00, 19.99it/s]

Test set: Average loss: 0.6631, Accuracy: 7727/10000 (77.27%)

validation-accuracy improved from 76.91 to 77.27, saving model to D:\PG-ML\eva4\week7\saved_models\CIFAR10_model_epoch-10_L1-1_L2-0_val_acc-77.27.h5
EPOCH: 11
Loss=0.7235488295555115 Batch_id=781 Accuracy=79.60: 100%|███████████████████████████| 782/782 [00:39<00:00, 19.93it/s]

Test set: Average loss: 0.6799, Accuracy: 7635/10000 (76.35%)

EPOCH: 12
Loss=0.7278296947479248 Batch_id=781 Accuracy=80.31: 100%|███████████████████████████| 782/782 [00:40<00:00, 19.54it/s]

Test set: Average loss: 0.6477, Accuracy: 7775/10000 (77.75%)

validation-accuracy improved from 77.27 to 77.75, saving model to D:\PG-ML\eva4\week7\saved_models\CIFAR10_model_epoch-12_L1-1_L2-0_val_acc-77.75.h5
EPOCH: 13
Loss=0.8402858376502991 Batch_id=781 Accuracy=80.87: 100%|███████████████████████████| 782/782 [00:42<00:00, 18.23it/s]

Test set: Average loss: 0.6721, Accuracy: 7674/10000 (76.74%)

EPOCH: 14
Loss=0.4658941924571991 Batch_id=781 Accuracy=81.27: 100%|███████████████████████████| 782/782 [00:39<00:00, 19.97it/s]

Test set: Average loss: 0.6301, Accuracy: 7800/10000 (78.00%)

validation-accuracy improved from 77.75 to 78.0, saving model to D:\PG-ML\eva4\week7\saved_models\CIFAR10_model_epoch-14_L1-1_L2-0_val_acc-78.0.h5
EPOCH: 15
Loss=0.36543208360671997 Batch_id=781 Accuracy=81.83: 100%|██████████████████████████| 782/782 [00:39<00:00, 20.00it/s]

Test set: Average loss: 0.6987, Accuracy: 7584/10000 (75.84%)

EPOCH: 16
Loss=0.2142392098903656 Batch_id=781 Accuracy=81.95: 100%|███████████████████████████| 782/782 [00:39<00:00, 19.94it/s]

Test set: Average loss: 0.6727, Accuracy: 7740/10000 (77.40%)

EPOCH: 17
Loss=0.4537639021873474 Batch_id=781 Accuracy=82.73: 100%|███████████████████████████| 782/782 [00:39<00:00, 19.95it/s]

Test set: Average loss: 0.7033, Accuracy: 7661/10000 (76.61%)

EPOCH: 18
Loss=0.5358160734176636 Batch_id=781 Accuracy=82.92: 100%|███████████████████████████| 782/782 [00:39<00:00, 19.86it/s]

Test set: Average loss: 0.5811, Accuracy: 7983/10000 (79.83%)

validation-accuracy improved from 78.0 to 79.83, saving model to D:\PG-ML\eva4\week7\saved_models\CIFAR10_model_epoch-18_L1-1_L2-0_val_acc-79.83.h5
EPOCH: 19
Loss=0.3986873924732208 Batch_id=781 Accuracy=83.45: 100%|███████████████████████████| 782/782 [00:39<00:00, 19.78it/s]

Test set: Average loss: 0.5961, Accuracy: 7940/10000 (79.40%)

EPOCH: 20
Loss=1.1048436164855957 Batch_id=781 Accuracy=83.58: 100%|███████████████████████████| 782/782 [00:39<00:00, 19.84it/s]

Test set: Average loss: 0.6449, Accuracy: 7857/10000 (78.57%)

EPOCH: 21
Loss=0.512848973274231 Batch_id=781 Accuracy=83.86: 100%|████████████████████████████| 782/782 [00:39<00:00, 19.88it/s]

Test set: Average loss: 0.6092, Accuracy: 7913/10000 (79.13%)

EPOCH: 22
Loss=0.41277021169662476 Batch_id=781 Accuracy=84.21: 100%|██████████████████████████| 782/782 [00:39<00:00, 19.92it/s]

Test set: Average loss: 0.5937, Accuracy: 7988/10000 (79.88%)

validation-accuracy improved from 79.83 to 79.88, saving model to D:\PG-ML\eva4\week7\saved_models\CIFAR10_model_epoch-22_L1-1_L2-0_val_acc-79.88.h5
EPOCH: 23
Loss=0.35014504194259644 Batch_id=781 Accuracy=84.58: 100%|██████████████████████████| 782/782 [00:39<00:00, 19.82it/s]

Test set: Average loss: 0.6252, Accuracy: 7883/10000 (78.83%)

EPOCH: 24
Loss=0.343058705329895 Batch_id=781 Accuracy=84.77: 100%|████████████████████████████| 782/782 [00:39<00:00, 19.83it/s]

Test set: Average loss: 0.5764, Accuracy: 8083/10000 (80.83%)

validation-accuracy improved from 79.88 to 80.83, saving model to D:\PG-ML\eva4\week7\saved_models\CIFAR10_model_epoch-24_L1-1_L2-0_val_acc-80.83.h5
EPOCH: 25
Loss=0.3980933129787445 Batch_id=781 Accuracy=84.78: 100%|███████████████████████████| 782/782 [00:40<00:00, 19.55it/s]

Test set: Average loss: 0.5948, Accuracy: 7981/10000 (79.81%)

EPOCH: 26
Loss=0.9966412782669067 Batch_id=781 Accuracy=85.47: 100%|███████████████████████████| 782/782 [00:39<00:00, 19.78it/s]

Test set: Average loss: 0.6166, Accuracy: 7919/10000 (79.19%)

EPOCH: 27
Loss=1.188391089439392 Batch_id=781 Accuracy=85.31: 100%|████████████████████████████| 782/782 [00:39<00:00, 19.86it/s]

Test set: Average loss: 0.5922, Accuracy: 8006/10000 (80.06%)

EPOCH: 28
Loss=0.35659366846084595 Batch_id=781 Accuracy=85.63: 100%|██████████████████████████| 782/782 [00:39<00:00, 19.91it/s]

Test set: Average loss: 0.5783, Accuracy: 8063/10000 (80.63%)

EPOCH: 29
Loss=0.6107515096664429 Batch_id=781 Accuracy=86.10: 100%|███████████████████████████| 782/782 [00:39<00:00, 19.77it/s]

Test set: Average loss: 0.6197, Accuracy: 7921/10000 (79.21%)

EPOCH: 30
Loss=0.483569860458374 Batch_id=781 Accuracy=85.95: 100%|████████████████████████████| 782/782 [00:39<00:00, 19.75it/s]

Test set: Average loss: 0.5927, Accuracy: 8008/10000 (80.08%)

EPOCH: 31
Loss=0.7053028345108032 Batch_id=781 Accuracy=86.07: 100%|███████████████████████████| 782/782 [00:39<00:00, 19.95it/s]

Test set: Average loss: 0.5860, Accuracy: 8067/10000 (80.67%)

EPOCH: 32
Loss=0.474185973405838 Batch_id=781 Accuracy=86.68: 100%|████████████████████████████| 782/782 [00:39<00:00, 19.95it/s]

Test set: Average loss: 0.5967, Accuracy: 8008/10000 (80.08%)

EPOCH: 33
Loss=0.6773715615272522 Batch_id=781 Accuracy=86.54: 100%|███████████████████████████| 782/782 [00:39<00:00, 19.97it/s]

Test set: Average loss: 0.5843, Accuracy: 8053/10000 (80.53%)

EPOCH: 34
Loss=0.7049911618232727 Batch_id=781 Accuracy=86.65: 100%|███████████████████████████| 782/782 [00:39<00:00, 19.91it/s]

Test set: Average loss: 0.5671, Accuracy: 8076/10000 (80.76%)

EPOCH: 35
Loss=0.5176164507865906 Batch_id=781 Accuracy=86.62: 100%|███████████████████████████| 782/782 [00:39<00:00, 19.72it/s]

Test set: Average loss: 0.5957, Accuracy: 8031/10000 (80.31%)

EPOCH: 36
Loss=0.3645068109035492 Batch_id=781 Accuracy=87.00: 100%|███████████████████████████| 782/782 [00:39<00:00, 19.79it/s]

Test set: Average loss: 0.6017, Accuracy: 8009/10000 (80.09%)

EPOCH: 37
Loss=1.0083394050598145 Batch_id=781 Accuracy=87.02: 100%|███████████████████████████| 782/782 [00:39<00:00, 19.93it/s]

Test set: Average loss: 0.5681, Accuracy: 8115/10000 (81.15%)

validation-accuracy improved from 80.83 to 81.15, saving model to D:\PG-ML\eva4\week7\saved_models\CIFAR10_model_epoch-37_L1-1_L2-0_val_acc-81.15.h5
EPOCH: 38
Loss=0.5425737500190735 Batch_id=781 Accuracy=87.20: 100%|███████████████████████████| 782/782 [00:39<00:00, 19.93it/s]

Test set: Average loss: 0.5932, Accuracy: 8075/10000 (80.75%)

EPOCH: 39
Loss=0.3140729069709778 Batch_id=781 Accuracy=87.59: 100%|███████████████████████████| 782/782 [00:39<00:00, 19.91it/s]

Test set: Average loss: 0.6103, Accuracy: 7999/10000 (79.99%)

EPOCH: 40
Loss=0.8436259031295776 Batch_id=781 Accuracy=87.28: 100%|███████████████████████████| 782/782 [00:39<00:00, 19.89it/s]

Test set: Average loss: 0.6315, Accuracy: 7942/10000 (79.42%)

```
Here are the test-loss and test-accuracy plots over a 40-epoch-run:

![alt text](https://github.com/ojhajayant/eva/blob/master/week7/test_loss_acc.PNG "Logo Title Text 1")

Here are example confusion-matrix and classification-report:

![alt text](https://github.com/ojhajayant/eva/blob/master/week7/classification_matrix.PNG "Logo Title Text 1")

Here are 3 groups of mislabelled 10-class samples from the test-dataset:

![alt text](https://github.com/ojhajayant/eva/blob/master/week7/mislabelled_cifar.PNG "Logo Title Text 1")

Here are [example train-cmd run-log](https://github.com/ojhajayant/eva/blob/master/week7/an_example_train_cmd_run.log) & [example test-cmd run-log](https://github.com/ojhajayant/eva/blob/master/week7/an_example_train_cmd_run.log)

      
While the overall code is structured as demanded, but some of initial experimental results can be found 
 @ the [notebook](https://github.com/ojhajayant/eva/blob/master/week7/S6_assignment_expt_01_with_depthwise_separable_conv2d_02.ipynb) & also  [here](https://github.com/ojhajayant/eva/blob/master/week7/S6_assignment_expt_01_with_depthwise_separable_conv2d_01.ipynb) &  [here](https://github.com/ojhajayant/eva/blob/master/week7/S6_assignment_expt_01.ipynb)
 
