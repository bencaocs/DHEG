import os
import pandas as pd
from torch.autograd import Variable
from sklearn import metrics
from AGATPPIS_model import *
import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt
import time
# Path
Dataset_Path = "/home/bio-3090ti/BenzcodeL/MEGPPIS/Dataset/"
# # Model_Path = "./Model/2022-09-06-11-49-17/model/" orignal model
# # Model_Path = "./Log/2023-09-23-14-09-48/model/" egnn 3 model
# # Model_Path = "./Log/2023-09-25-00-02-43/model/" # sub egnn 2  model 1 0.7 0.5 
# # Model_Path = "./Log/2023-09-25-16-44-00/model/" # sub egnn 3  model 1 0.7 0.5 
# # Model_Path = "./Log/2023-09-25-17-14-48/model/" # egnn 6  model 
# # Model_Path = "./Log/2023-09-25-22-24-50/model/" #  u egnn 0.9 0.7  model 2023-09-25-20-45-51
# # Model_Path = "./Log/2023-09-26-08-42-55/model/" # egnn sub 1
# # Model_Path = "./Log/2023-09-26-20-25-29/model/" # egnn sub 0.7
# # Model_Path = "./Log/2023-09-27-22-04-44/model/" # egnn sub 1 0.7
# # Model_Path = "./Log/2023-09-27-22-54-39/model/" # egnn sub 1 0.7 0.5 70
# # Model_Path = "./Log/2023-09-28-21-15-13/model/" # egnn sub 1 0.9
# # Model_Path = "./Log/2023-09-28-21-40-24/model/" # egnn sub 1 0.8
# # Model_Path = "./Log/2023-09-28-22-40-16/model/" # egnn sub 1 0.6
# # Model_Path = "./Log/2023-10-07-09-49-28/model/" # egnn sub 1 0.7 + atom 2023-10-07-10-11-33
# # Model_Path = "./Log/2023-10-07-11-48-55/model/" # egnn sub 1 0.7 + atom2 2023-10-07-11-48-55
# # Model_Path = "./Log/2023-10-07-12-05-25/model/"# egnn sub 1 2023-10-07-12-28-48
# # Model_Path = "./Log/2023-10-07-12-28-48/model/"# egnn sub 1 0.75 all 20
# # Model_Path = "./Log/2023-10-07-15-24-12/model/"# egnn sub 1 0.75 all 24 2023-10-07-13-22-43 22 2023-10-07-15-24-12
# # Model_Path = "./Log/2023-10-07-15-45-12/model/"# egnn sub 1 0.75 all 18 drop 0.1
# # Model_Path = "./Log/2023-10-07-16-12-08/model/"# egnn sub 1 0.7 all 20
# # Model_Path = "./Log/2023-10-07-16-28-01/model/"# egnn sub 1 0.8 all 20
# # Model_Path = "./Log/2023-10-07-16-48-11/model/"# egnn sub 1 0.75 all 20 5fold
# # Model_Path = "./Log/2023-10-07-23-50-20/model/" # egnn sub 1 0.75 all 26
# # Model_Path = "./Log/2023-10-08-00-17-19/model/" # egnn sub 1 0.75 all  linear: 10 10
# # Model_Path = "./Log/2023-10-08-00-37-28/model/" #egnn sub 1 0.75 all linear 20 10 0.7 0.3
# # Model_Path = "./Log/2023-10-08-10-44-34/model/" # egnn sub 1 0.75 all linear 20 10 epoch 60 seed 27
# # Model_Path = "./Log/2023-10-08-11-14-42/model/" # egnn sub 1 0.75 all linear 20 10 epoch 60 seed 2020
# # Model_Path = "./Log/2023-10-08-14-57-04/model/" #  egnn sub 1 0.75 all linear 256 epoch 50 seed 3407
# # Model_Path = "./Log/2023-10-08-15-22-21/model/" # egnn sub 1 0.75 all linear 20 20 epoch 50 seed 3407
# # Model_Path = "./Log/2023-10-08-16-48-38/model/"  # egnn sub layer 4  1 0.75 all 16 linear 20 10 epoch 50 seed 2020 best 
# # Model_Path = "./Log/2023-10-08-17-18-36/model/" # egnn sub layer 4  1 0.75 all 16 linear 20 10 epoch 50 seed 2020 5 fold
# # Model_Path = "./Log/2023-10-09-10-57-27/model/" # egnn sub layer 4  1 0.75 16 all linear 20 10 epoch 60 seed 2020 best

# # Model_Path = "./Log/2023-10-09-11-27-42/model/" # egnn sub layer 4  1 0.75 all 17  linear 20 10 epoch 60 seed 2020 best m17
# # Model_Path = "./Log/2023-10-09-15-12-42/model/" # egnn sub layer 4  1 0.75 +atom 17  linear 20 10 epoch 60 seed 2020 best m17
# # Model_Path = "./Log/2023-10-09-15-37-25/model/" # egnn sub layer 4  1  all 17  linear 20 10 epoch 60 seed 2020 best m17
# # Model_Path = "./Log/2023-10-09-16-00-33/model/" # egnn sub layer 4  0.75  all 17  linear 20 10 epoch 60 seed 2020 best m172023-10-09-16-00-33
# # Model_Path = "./Log/test_model_1/model/"
# # Model_Path = "./Log/agat_model/model/"
# # 2023-10-12-19-58-40 all
# # 2023-10-12-22-00-16 -atom 
# # Model_Path = "./Log/2023-10-23-13-28-55/model/" # 0.806 0.473
# # 2023-10-13-09-17-23 -pose
# # Model_Path = "./Log/2023-10-23-13-50-35/model/" # 0.848 0.555
# # 2023-10-13-15-11-42 -atom+pose  
# # Model_Path = "./Log/2023-10-23-14-11-05/model/" # 0.805 0.470 
# # 2023-10-13-16-48-22 1
# # Model_Path = "./Log/2023-10-23-15-49-23/model/" # 0.819 0.486
# # 2023-10-13-18-53-10 0.75
# # Model_Path = "./Log/2023-10-23-16-07-05/model/" # 0.778 0.441
# # 2023-10-13-21-41-49 CA 
# # Model_Path = "./Log/2023-10-23-14-32-51/model/"  # 0.849 0.542
# # 2023-10-13-23-24-30 C 
# # Model_Path = "./Log/2023-10-23-15-26-10/model/" # 0.847 0.549
# # GCN 0.7942 0.4522
# # Model_Path = "./Log/2023-10-23-16-27-08/model/" # 0.803 0.457
# # GAT 0.8152 0.4847
# # Model_Path = "./Log/2023-10-24-10-43-18/model1/" # 0.827 0.501
# # Model_Path = "./Log/2023-10-23-19-51-42/model/" # 0.831 0.496

# Model_Path = "./Log/2023-11-02-12-48-55/model/" # MAP_CUTOFF = 13 

# Model_Path = "./Log/2024-01-22-03-15-25/model/" # 1 0.75
# Model_Path = "./Log/2024-01-22-03-30-31/model/" # 1

# Model_Path = "./Log/2024-01-22-03-38-53/model/" # +** 1 0.75 4 layer

# Model_Path = "./Log/2024-01-22-03-54-12/model/" # +** 1 0.75 5 layer

# Model_Path = "./Log/2024-01-22-04-08-23/model/" # +** 1 0.75 3 layer

# # Model_Path = "./Log/2024-01-22-04-17-46/model/" # +** 1 0.7

# # Model_Path = "./Log/2024-01-22-04-26-51/model/" # +** 1 0.75 +norm

# # Model_Path = "./Log/2024-01-22-04-26-51/model/" # +** 1 0.75  5fold


# # Model_Path = "./Log/2024-01-22-03-38-53/model/" # +** 1 0.75 4 layer

# # Model_Path = "./Log/2024-01-22-05-28-02/model/" # +** 1 0.8 4 layer

# # Model_Path = "./Log/2024-01-22-05-41-46/model/" # +** 1 0.85 4 layer

# # Model_Path = "./Log/2024-01-22-05-11-06/model/" # +** 1  4 layer 45

# # Model_Path = "./Log/2024-01-22-05-17-50/model/" # +** 1  4 layer 30 

# # +AF /home/hy/MEG/code/Log/2024-01-22-05-55-32

# # Model_Path = "./Log/2024-01-22-05-55-32/model/"
# #
# # Full_model_47.pkl
# # 1.0970375537872314
# # ========== Evaluate Test set ==========
# # Test loss:  0.3586196221411228
# # Test binary acc:  0.8557516737674985
# # Test precision: 0.5372761349437735
# # Test recall:  0.6216867469879518
# # Test f1:  0.5764075067024129
# # Test AUC:  0.8690870737444311
# # Test AUPRC:  0.6035656817703123
# # Test mcc:  0.491932727606243
# # Threshold:  0.34

# # +PEF /home/hy/MEG/code/Log/2024-01-22-05-57-25

# # Model_Path = "./Log/2024-01-22-05-57-25/model/"
# # Full_model_50.pkl
# # 1.080256700515747
# # ========== Evaluate Test set ==========
# # Test loss:  0.38463891098896663
# # Test binary acc:  0.832242848447961
# # Test precision: 0.475
# # Test recall:  0.5951807228915663
# # Test f1:  0.5283422459893049
# # Test AUC:  0.8433943706890076
# # Test AUPRC:  0.5392109842382684
# # Test mcc:  0.43190687622328694
# # Threshold:  0.22

# # no /home/hy/MEG/code/Log/2024-01-22-06-08-50
# # Model_Path = "./Log/2024-01-22-06-08-50/model/"

# # # CA /home/hy/MEG/code/Log/2024-01-22-06-19-42
# # # 2024-01-22-06-41-49
# # Model_Path = "./Log/2024-01-22-06-48-56/model/"

# # # C 2024-01-22-06-42-59
# # # 2024-01-22-06-47-37 1 0.75
# # # 2024-01-22-14-53-48 1 0.8
# # Model_Path = "./Log/2024-01-22-14-53-48/model/"

# # 1e-4 /home/hy/MEG/code/Log/2024-01-22-06-32-19
# # Model_Path = "./Log/2024-01-22-06-32-19/model/"
# # Full_model_49.pkl
# # 1.088071584701538
# # ========== Evaluate Test set ==========
# # Test loss:  0.3868181504309177
# # Test binary acc:  0.8043974437005478
# # Test precision: 0.42041078305519897
# # Test recall:  0.6313253012048192
# # Test f1:  0.5047197071855135
# # Test AUC:  0.8303480576928728
# # Test AUPRC:  0.5370606042171051
# # Test mcc:  0.4013821834452608
# # Threshold:  0.21



# # Full_model_46.pkl
# # 1.0635936260223389
# # ========== Evaluate Test set ==========
# # Test loss:  0.36924497882525126
# # Test binary acc:  0.8685331710286062
# # Test precision: 0.5760631302060499
# # Test recall:  0.6332530120481927
# # Test f1:  0.6033057851239669
# # Test AUC:  0.8854148620863433
# # Test AUPRC:  0.6325560827059467
# # Test mcc:  0.525572184929143


# # cutoff 13 2024-01-22-15-19-23
# # Model_Path = "./Log/2024-01-22-15-19-23/model/"

# # cutoff 15 2024-01-22-15-19-23
# # Model_Path = "./Log/2024-01-22-15-39-14/model/"
# # Model_Path = "./Log/2024-01-22-15-48-03/model/"

# # # 1 1-50 /home/hy/MEG/code/Log/2024-01-22-16-14-11

# # Model_Path = "./Log/2024-01-22-16-14-11/model/"

# # # 1 1-50 5 layer 2024-01-22-16-22-26

# Model_Path = "./Log/2024-01-22-16-22-26/model/"



# # # +** 1 0.7 5 layer 1-50 2024-01-22-17-00-36
# # Model_Path = "./Log/2024-01-22-17-00-36/model/"

# # Model_Path = "./Log/2024-01-22-17-12-03/model/"

# # Model_Path = "./Log/2024-01-22-16-14-11/model/"
# # Full_model_43.pkl
# # 1.1247718334197998
# # ========== Evaluate Test set ==========
# # Test loss:  0.32836134483416873
# # Test binary acc:  0.8742391965916008
# # Test precision: 0.599247412982126
# # Test recall:  0.6139759036144579
# # Test f1:  0.6065222566055702
# # Test AUC:  0.8877707523562495
# # Test AUPRC:  0.6530985872557773
# # Test mcc:  0.5317504878292423
# # Threshold:  0.33
# # Full_model_44.pkl
# # 1.0824520587921143
# # ========== Evaluate Test set ==========
# # Test loss:  0.36968769940237206
# # Test binary acc:  0.8704351795496044
# # Test precision: 0.5863509749303621
# # Test recall:  0.6086746987951808
# # Test f1:  0.5973043272641287
# # Test AUC:  0.8811434952929434
# # Test AUPRC:  0.6457726781784202
# # Test mcc:  0.5202632988488735

# # 1 1-50 5 layer /home/hy/MEG/code/Log/2024-01-22-18-07-09 p=0.3

# # Model_Path = "./Log/2024-01-22-18-07-09/model/"
# # # 1 0.5  5 layer
# # Model_Path = "./Log/2024-01-22-18-32-30/model/"

# #  # +** 1 0.75 5 layer 1-50
# # Model_Path = "./Log/2024-01-22-16-31-22/model/"
# # Full_model_43.pkl
# # 1.115696668624878
# # ========== Evaluate Test set ==========
# # Test loss:  0.32836134483416873
# # Test binary acc:  0.8742391965916008
# # Test precision: 0.599247412982126
# # Test recall:  0.6139759036144579
# # Test f1:  0.6065222566055702
# # Test AUC:  0.8877707523562495
# # Test AUPRC:  0.6530985872557773
# # Test mcc:  0.5317504878292423
# # Threshold:  0.33

# # #  # +** 1  6 layer 1-50
# # Model_Path = "./Log/2024-01-22-18-47-41/model/"

# # print(Model_Path[0:-6])
# # exit()
# # Full_model_38.pkl
# # 1.0619301795959473
# # ========== Evaluate Test set ==========
# # Test loss:  0.3448850451658169
# # Test binary acc:  0.8695982958003652
# # Test precision: 0.5791319596668129
# # Test recall:  0.6366265060240964
# # Test f1:  0.6065197428833793
# # Test AUC:  0.8908369733337542
# # Test AUPRC:  0.6381613064692401
# # Test mcc:  0.5294289644621664
# # Threshold:  0.24


#  # +** 1 0.6  6 layer 1-50 best
# # Model_Path = "./Log/2024-01-22-19-01-23/model/"
# # Test loss:  0.3362689637889465
# # Test binary acc:  0.878119293974437
# # Test precision: 0.60497114957834
# # Test recall:  0.656867469879518
# # Test f1:  0.6298521256931608
# # Test AUC:  0.8922070865447516
# # Test AUPRC:  0.66556795257964
# # Test mcc:  0.557724866196188

# # # #  # +** 1 0.65  5 layer 1-50 best
# # # Model_Path = "./Log/2024-01-22-19-21-05/model/"

# # # #  # +** 1 0.55  5 layer 1-50 best
# # # Model_Path = "./Log/2024-01-22-19-31-05/model/"

# # # #  # +** 1   7 layer 1-50 best
# # # Model_Path = "./Log/2024-01-22-19-48-19/model/"

# # #  # +** 1 0.72   5 layer 1-50 best
# # # Model_Path = "./Log/2024-01-22-20-38-28/model/"

# # #  # +** 1 0.75   4 layer 1-50 
# # Model_Path = "./Log/2024-01-22-22-39-07/model/"

# # Full_model_43.pkl
# # 1.0891728401184082
# # ========== Evaluate Test set ==========
# # Test loss:  0.34464777720471224
# # Test binary acc:  0.8654899573950091
# # Test precision: 0.5687416032243618
# # Test recall:  0.6120481927710844
# # Test f1:  0.5896007428040854
# # Test AUC:  0.8869007659511476
# # Test AUPRC:  0.6364746892499784
# # Test mcc:  0.5097837609178557
# # Threshold:  0.23

# #  # +** 1    4 layer 1-50 
# # Model_Path = "./Log/2024-01-22-22-48-22/model/"

# #  # +** 1    2 layer 1-50 
# # Model_Path = "./Log/2024-01-22-23-07-02/model/"

# #  # +** 1 0.6   2 layer 1-50 
# # Model_Path = "./Log/2024-01-22-23-19-45/model/"

# # # /home/hy/MEG/code/Log/2024-01-23-07-25-51
# #  # +** 1   3 layer 1-50 
# # Model_Path = "./Log/2024-01-23-07-25-51/model/"
# # Model_Path = "./Log/2024-01-23-07-38-52/model/"

# #  # +** 1  1 layer 1-50 
# # Model_Path = "./Log/2024-01-23-07-51-51/model/"

# #  # +** 1    2 layer 1-50 
# # Model_Path = "./Log/2024-01-23-08-03-58/model/"

# #  # +** 1 0.75   3 layer 1-50 
# # Model_Path = "./Log/2024-01-23-08-20-31/model/"
# # Full_model_35.pkl
# # 1.010286808013916
# # ========== Evaluate Test set ==========
# # Test loss:  0.3628832531472047
# # Test binary acc:  0.8579580036518564
# # Test precision: 0.5441051738761662
# # Test recall:  0.6183132530120482
# # Test f1:  0.5788405143243853
# # Test AUC:  0.8682951736478844
# # Test AUPRC:  0.606670039613185
# # Test mcc:  0.4952922390143642
# # Threshold:  0.24

#  # +** 1 0.6   6 layer 1-50  atom
# # Model_Path = "./Log/2024-01-23-09-24-10/model/"
# # ========== Evaluate Test set ==========
# # Test loss:  0.3533981499572595
# # Test binary acc:  0.8734783931832014
# # Test precision: 0.5968045112781954
# # Test recall:  0.6120481927710844
# # Test f1:  0.6043302403045443
# # Test AUC:  0.8849885983540268
# # Test AUPRC:  0.6298902173414597
# # Test mcc:  0.5291041171491366
# # Threshold:  0.26

#  # +** 1 0.6   6 layer 1-50  pos
# # Model_Path = "./Log/2024-01-23-09-36-03/model/"
# # Full_model_46.pkl
# # 1.0943081378936768
# # ========== Evaluate Test set ==========
# # Test loss:  0.3947608437389135
# # Test binary acc:  0.8422093730979915
# # Test precision: 0.5001912045889101
# # Test recall:  0.6303614457831326
# # Test f1:  0.5577825159914712
# # Test AUC:  0.8480088644395996
# # Test AUPRC:  0.5644751932053584
# # Test mcc:  0.46789086808719116
# # Threshold:  0.15

#  # +** 1 0.6   6 layer 1-50  no
# # Model_Path = "./Log/2024-01-23-10-01-42/model/" # code/Log/2024-01-23-10-01-42

# # Full_model_41.pkl
# # 1.1378173828125
# # ========== Evaluate Test set ==========
# # Test loss:  0.37975120916962624
# # Test binary acc:  0.8412964090079124
# # Test precision: 0.49797421731123387
# # Test recall:  0.651566265060241
# # Test f1:  0.5645093945720251
# # Test AUC:  0.8618795137184387
# # Test AUPRC:  0.5843276342904408
# # Test mcc:  0.47593112887527295
# # Threshold:  0.19

#  # +** 1 0.6   6 layer 1-50  -no - dssp
# # Model_Path = "./Log/2024-01-23-11-19-00/model/" 
# # Full_model_37.pkl
# # 1.1507682800292969
# # ========== Evaluate Test set ==========
# # Test loss:  0.38360156963268915
# # Test binary acc:  0.829199634814364
# # Test precision: 0.4691582002902758
# # Test recall:  0.623132530120482
# # Test f1:  0.5352929000206996
# # Test AUC:  0.8363064980130115
# # Test AUPRC:  0.549565921902471
# # Test mcc:  0.4397493384578476




# # Model_Path = "./Model/test_model/"
# # Model_Path = "./Model/test_model/model3/"
# Model_Path = "./Model/agat_model/model1/"


# # # +**    6 layer 1-50 --- 1 0.62 1 0.55 
# # Model_Path = "./Log/2024-01-24-22-33-09/model/" 


# # +** 1 0.58   6 layer 1-50 
# # Test loss:  0.3254157560567061
# # Test binary acc:  0.8742391965916008
# # Test precision: 0.5923009623797025
# # Test recall:  0.6525301204819277
# # Test f1:  0.6209584957578537
# # Test AUC:  0.8961835235058946
# # Test AUPRC:  0.674698188563278
# # Test mcc:  0.5467031328484978

# # Model_Path = "./Log/2024-01-24-22-47-19/model/" 
# # # 1 0.61
# # Model_Path = "./Log/2024-01-24-23-04-14/model/" 
# # Full_model_46.pkl
# # 1.1241810321807861
# # ========== Evaluate Test set ==========
# # Test loss:  0.3368488933891058
# # Test binary acc:  0.8728697504564821
# # Test precision: 0.5849453322119428
# # Test recall:  0.6703614457831325
# # Test f1:  0.6247473613294408
# # Test AUC:  0.8903267020562148
# # Test AUPRC:  0.6534502254105565
# # Test mcc:  0.5504918319008582
# # Threshold:  0.25

# # 1 0.5
# # Model_Path = "./Log/2024-01-24-23-17-27/model/" 
# # Full_model_43.pkl
# # 1.1786730289459229
# # ========== Evaluate Test set ==========
# # Test loss:  0.35878642201423644
# # Test binary acc:  0.8751521606816799
# # Test precision: 0.5991773308957953
# # Test recall:  0.6318072289156627
# # Test f1:  0.6150598170302604
# # Test AUC:  0.8887156032205432
# # Test AUPRC:  0.6537781137858893
# # Test mcc:  0.5408867410370634
# # Threshold:  0.25

# # 1 0.55
# # Model_Path = "./Log/2024-01-24-23-54-20/model/" 
# # Full_model_47.pkl
# # 1.1079134941101074
# # ========== Evaluate Test set ==========
# # Test loss:  0.33884830996394155
# # Test binary acc:  0.8772824102251978
# # Test precision: 0.611810261374637
# # Test recall:  0.6091566265060241
# # Test f1:  0.610480560251147
# # Test AUC:  0.884339156245544
# # Test AUPRC:  0.6425541621308322
# # Test mcc:  0.5376512229423129
# # Threshold:  0.34
# # 1 0.7 
# # Full_model_45.pkl
# # 1.082679033279419
# # ========== Evaluate Test set ==========
# # Test loss:  0.33114705060919125
# # Test binary acc:  0.8797930614729154
# # Test precision: 0.6180257510729614
# # Test recall:  0.624578313253012
# # Test f1:  0.6212847555129435
# # Test AUC:  0.8909004960124173
# # Test AUPRC:  0.6645654299006517
# # Test mcc:  0.5498564278570919
# # Model_Path = "./Log/2024-01-25-00-05-37/model/" 

# # +** 1 0.65   6 layer 1-50 
# # Model_Path = "./Log/2024-01-24-22-33-09/model/" 

# # Full_model_39.pkl
# # 1.2035152912139893
# # ========== Evaluate Test set ==========
# # Test loss:  0.34052263423800466
# # Test binary acc:  0.8675441265976872
# # Test precision: 0.5707627118644067
# # Test recall:  0.6491566265060241
# # Test f1:  0.6074408117249155
# # Test AUC:  0.8870373680103012
# # Test AUPRC:  0.6272010252408242
# # Test mcc:  0.5297501014152097
# # Threshold:  0.36
# # Model_Path = "./Model/test_model/model1/"

# # # +** 1    6 layer 1-50 no  0.52
# # Model_Path = "./Log/2024-01-25-21-16-49/model/" 

# # # +** 1   2 layer 1-50 no  0.787 0.439
# # Model_Path = "./Log/2024-01-25-21-25-07/model/" 

# # # +** 1   5 layer 1-50 no  0.850 0.548
# # Model_Path = "./Log/2024-01-25-21-32-03/model/" 

# # # +** 1   4 layer 1-50 no  0.832 0.563
# # Model_Path = "./Log/2024-01-25-21-43-34/model/" 

# # # +** 1   3 layer 1-50 no  0.825 0.527
# # Model_Path = "./Log/2024-01-25-21-52-59/model/" 

# # # +** 1 0.6   5 layer 1-50 no  0.856 0.572
# # Model_Path = "./Log/2024-01-25-22-27-31/model/" 

# # # +** 1 0.6   4 layer 1-50 no  0.8 0.51
# # Model_Path = "./Log/2024-01-25-22-46-51/model/" 

# # # +** 1 0.5   4 layer 1-50 no  0.840 0.544
# # Model_Path = "./Log/2024-01-25-22-56-19/model/" 

# # # +** 1 0.7   4 layer 1-50 no  0.857 0.563
# # Model_Path = "./Log/2024-01-25-23-11-23/model/" 

# # +** 1 0.4   6 layer 1-50   
# Model_Path = "./Log/2024-01-26-22-48-29/model/" 
# # Full_model_45.pkl
# # 0.9923262596130371
# # ========== Evaluate Test set ==========
# # Test loss:  0.36226124577224256
# # Test binary acc:  0.8589470480827754
# # Test precision: 0.5458696554586966
# # Test recall:  0.6337349397590362
# # Test f1:  0.5865298840321143
# # Test AUC:  0.8791650838606028
# # Test AUPRC:  0.6270843100080359
# # Test mcc:  0.5041000115152379
# # +** 1 0.3   6 layer 1-50   
# Model_Path = "./Log/2024-01-26-22-59-05/model/" 
# # Full_model_48.pkl
# # 1.094780445098877
# # ========== Evaluate Test set ==========
# # Test loss:  0.34056415731708206
# # Test binary acc:  0.8738587948874011
# # Test precision: 0.6018563751831949
# # Test recall:  0.5937349397590361
# # Test f1:  0.5977680737506065
# # Test AUC:  0.8761019323476942
# # Test AUPRC:  0.6360657688397133
# # Test mcc:  0.52298989571887
# # +** 1 0.2   6 layer 1-50   
# Model_Path = "./Log/2024-01-26-23-10-27/model/" 
# # Full_model_50.pkl
# # 0.9916396141052246
# # ========== Evaluate Test set ==========
# # Test loss:  0.35251371761163075
# # Test binary acc:  0.8696743761412051
# # Test precision: 0.5915065722952477
# # Test recall:  0.563855421686747
# # Test f1:  0.5773501110288676
# # Test AUC:  0.8691280652468034
# # Test AUPRC:  0.6154981537309989
# # Test mcc:  0.5005619555684494
# # Threshold:  0.27
# ## +** 1 0.1   6 layer 1-50   
# Model_Path = "./Log/2024-01-26-23-20-21/model/" 
# # Full_model_37.pkl
# # 1.035329818725586
# # ========== Evaluate Test set ==========
# # Test loss:  0.3376624492307504
# # Test binary acc:  0.8566646378575776
# # Test precision: 0.5358348968105066
# # Test recall:  0.6881927710843373
# # Test f1:  0.6025316455696202
# # Test AUC:  0.8909962589539656
# # Test AUPRC:  0.6491907197831042
# # Test mcc:  0.5227670944350962

# ## +** 1 0.1   5 layer 1-50   
# Model_Path = "./Log/2024-01-26-23-39-18/model/" 
# # Full_model_44.pkl
# # 1.080695390701294
# # ========== Evaluate Test set ==========
# # Test loss:  0.3532166020323833
# # Test binary acc:  0.8710438222763238
# # Test precision: 0.5862851952770209
# # Test recall:  0.6221686746987952
# # Test f1:  0.6036941781622632
# # Test AUC:  0.8884412235626034
# # Test AUPRC:  0.6547251683614262
# # Test mcc:  0.5270997526907282

# ## +** 1 0.2   5 layer 1-50   
# Model_Path = "./Log/2024-01-26-23-50-06/model/" 
# # Full_model_25.pkl
# # 1.0495648384094238
# # ========== Evaluate Test set ==========
# # Test loss:  0.3444734637935956
# # Test binary acc:  0.8646530736457699
# # Test precision: 0.5633019674935843
# # Test recall:  0.6346987951807229
# # Test f1:  0.5968728755948335
# # Test AUC:  0.8744643621010376
# # Test AUPRC:  0.6325326533776294
# # Test mcc:  0.5172203384047963

# ## +** 1 0.3   5 layer 1-50   2024-01-27-00-05-31
# Model_Path = "./Log/2024-01-27-00-08-37/model/" 
# # Full_model_37.pkl
# # 1.0368943214416504
# # ========== Evaluate Test set ==========
# # Test loss:  0.38032304296890895
# # Test binary acc:  0.85415398660986
# # Test precision: 0.5347100175746925
# # Test recall:  0.5865060240963855
# # Test f1:  0.5594116295104574
# # Test AUC:  0.8515411215736556
# # Test AUPRC:  0.5659050391079781
# # Test mcc:  0.4729730880783417
# # Threshold:  0.38

# ## +** 1 0.4   5 layer 1-50   
# Model_Path = "./Log/2024-01-27-00-18-53/model/" 
# # Full_model_21.pkl
# # 1.0816738605499268
# # ========== Evaluate Test set ==========
# # Test loss:  0.3642025776207447
# # Test binary acc:  0.8482958003651856
# # Test precision: 0.5155231889612879
# # Test recall:  0.6481927710843374
# # Test f1:  0.5742954739538857
# # Test AUC:  0.8760775943234497
# # Test AUPRC:  0.589542825368464
# # Test mcc:  0.4881464955527146

# ## +** 1 0.5   5 layer 1-50   
# Model_Path = "./Log/2024-01-27-00-23-35/model/" 
# # Full_model_47.pkl
# # 1.0968513488769531
# # ========== Evaluate Test set ==========
# # Test loss:  0.3546991697202126
# # Test binary acc:  0.8504260499087036
# # Test precision: 0.519926873857404
# # Test recall:  0.6853012048192771
# # Test f1:  0.5912681912681913
# # Test AUC:  0.8830683761334978
# # Test AUPRC:  0.6203850858693788
# # Test mcc:  0.509001678147414
# # Threshold:  0.27

# ## +** 1 0.6   5 layer 1-50   --
# Model_Path = "./Log/2024-01-27-00-33-39/model/" 

# ## +** 1 0.7   5 layer 1-50   
# Model_Path = "./Log/2024-01-27-00-35-47/model/" 
# # Full_model_50.pkl
# # 1.0529558658599854
# # ========== Evaluate Test set ==========
# # Test loss:  0.3513742653032144
# # Test binary acc:  0.8680006086427268
# # Test precision: 0.5728987993138936
# # Test recall:  0.643855421686747
# # Test f1:  0.6063081461311549
# # Test AUC:  0.8849620398660321
# # Test AUPRC:  0.6472587143069872
# # Test mcc:  0.5286366147952926
# # Threshold:  0.33

# ## +** 1    5 layer 1-50   

# Model_Path = "./Log/2024-01-22-16-22-26/model/"

# # Full_model_37.pkl
# # 1.0104382038116455
# # ========== Evaluate Test set ==========
# # Test loss:  0.34386080702145894
# # Test binary acc:  0.8700547778454047
# # Test precision: 0.579402855906534
# # Test recall:  0.6453012048192771
# # Test f1:  0.6105791153670771
# # Test AUC:  0.8943222306517606
# # Test AUPRC:  0.6479159632875072
# # Test mcc:  0.5339796798789546
# # Model_Path = "./Model/test_model/model1/"

# ## +** 1 0.1   4 layer 1-50   
# Model_Path = "./Log/2024-01-27-00-56-12/model/" 
# # Full_model_49.pkl
# # 1.1014773845672607
# # ========== Evaluate Test set ==========
# # Test loss:  0.3420664725204309
# # Test binary acc:  0.8767498478393183
# # Test precision: 0.6125680356259278
# # Test recall:  0.5966265060240964
# # Test f1:  0.6044921875000001
# # Test AUC:  0.8857182601578053
# # Test AUPRC:  0.643316649712656
# # Test mcc:  0.5315727594060516

# ## +** 1 0.2   4 layer 1-50   
# Model_Path = "./Log/2024-01-27-01-06-17/model/" 
# # Full_model_42.pkl
# # 1.2125475406646729
# # ========== Evaluate Test set ==========
# # Test loss:  0.36014947841564815
# # Test binary acc:  0.8666311625076081
# # Test precision: 0.5712389380530973
# # Test recall:  0.6221686746987952
# # Test f1:  0.5956170703575548
# # Test AUC:  0.8777203891906954
# # Test AUPRC:  0.6266327332548247
# # Test mcc:  0.5166132703401443
# # Threshold:  0.28

# ## +** 1 0.3   4 layer 1-50   
# Model_Path = "./Log/2024-01-27-01-12-21/model/" 
# # Full_model_43.pkl
# # 1.0258069038391113
# # ========== Evaluate Test set ==========
# # Test loss:  0.3542819092671076
# # Test binary acc:  0.8800213024954352
# # Test precision: 0.6299582463465553
# # Test recall:  0.5816867469879518
# # Test f1:  0.6048609371084941
# # Test AUC:  0.8842418694563238
# # Test AUPRC:  0.634981975715182
# # Test mcc:  0.5348562481676581
# # Threshold:  0.29

# ## +** 1 0.4   4 layer 1-50   
# Model_Path = "./Log/2024-01-27-01-21-07/model/" 
# # Full_model_28.pkl
# # 1.0916285514831543
# # ========== Evaluate Test set ==========
# # Test loss:  0.36470955858627957
# # Test binary acc:  0.8581101643335363
# # Test precision: 0.5406976744186046
# # Test recall:  0.672289156626506
# # Test f1:  0.5993555316863587
# # Test AUC:  0.8792324814662027
# # Test AUPRC:  0.6283217759037405
# # Test mcc:  0.5188801100946474

# ## +** 1 0.5   4 layer 1-50   
# Model_Path = "./Log/2024-01-27-01-27-17/model/" 
# # Full_model_43.pkl
# # 1.079530954360962
# # ========== Evaluate Test set ==========
# # Test loss:  0.3574099459995826
# # Test binary acc:  0.868837492391966
# # Test precision: 0.5793758480325645
# # Test recall:  0.6173493975903614
# # Test f1:  0.5977601493233785
# # Test AUC:  0.8889396959053124
# # Test AUPRC:  0.6483231645245263
# # Test mcc:  0.5198714892340667
# # Threshold:  0.25

# # ## +** 1 0.6   4 layer 1-50   
# Model_Path = "./Log/2024-01-27-01-28-09/model/" 
# # Full_model_44.pkl
# # 1.1186623573303223
# # ========== Evaluate Test set ==========
# # Test loss:  0.34250455535948277
# # Test binary acc:  0.8730219111381619
# # Test precision: 0.593376264949402
# # Test recall:  0.6216867469879518
# # Test f1:  0.6072016945163567
# # Test AUC:  0.8837219979384519
# # Test AUPRC:  0.6429064598943628
# # Test mcc:  0.5317266808185006
# # Threshold:  0.33
# #  # +** 1    4 layer 1-50 
# # Model_Path = "./Log/2024-01-22-22-48-22/model/"

# # ## +** 1 0.7   4 layer 1-50   
# Model_Path = "./Log/2024-01-27-01-55-28/model/" 
# # Full_model_36.pkl
# # 0.9850342273712158
# # ========== Evaluate Test set ==========
# # Test loss:  0.34355987012386324
# # Test binary acc:  0.8716524650030432
# # Test precision: 0.5931796349663785
# # Test recall:  0.5951807228915663
# # Test f1:  0.5941784941063267
# # Test AUC:  0.8807034298545705
# # Test AUPRC:  0.6322400826902902
# # Test mcc:  0.5179518713765087
# # Threshold:  0.3

# # ## +** 1 0.1   3 layer 1-50   
# Model_Path = "./Log/2024-01-27-02-05-28/model/" 
# # Full_model_35.pkl
# # 1.0907630920410156
# # ========== Evaluate Test set ==========
# # Test loss:  0.3576518436272939
# # Test binary acc:  0.8661746804625685
# # Test precision: 0.5715579710144928
# # Test recall:  0.6081927710843373
# # Test f1:  0.5893065608218537
# # Test AUC:  0.8722247196392401
# # Test AUPRC:  0.6262168365992854
# # Test mcc:  0.5098130399268155
# # Threshold:  0.29

# # ## +** 1 0.2   3 layer 1-50   
# Model_Path = "./Log/2024-01-27-02-13-01/model/" 
# # Full_model_49.pkl
# # 1.1258585453033447
# # ========== Evaluate Test set ==========
# # Test loss:  0.3766448823114236
# # Test binary acc:  0.8499695678636641
# # Test precision: 0.5221696082651743
# # Test recall:  0.584578313253012
# # Test f1:  0.5516143701682583
# # Test AUC:  0.858723538113063
# # Test AUPRC:  0.5629328237932884
# # Test mcc:  0.4629326821321938

# # ## +** 1 0.3   3 layer 1-50   
# Model_Path = "./Log/2024-01-27-02-18-31/model/" 
# # Full_model_39.pkl
# # 1.075596570968628
# # ========== Evaluate Test set ==========
# # Test loss:  0.37658492450912795
# # Test binary acc:  0.8344491783323189
# # Test precision: 0.48021934978456715
# # Test recall:  0.5908433734939759
# # Test f1:  0.5298184961106309
# # Test AUC:  0.8461696456074548
# # Test AUPRC:  0.5581134856183347
# # Test mcc:  0.43406269680653337

# ## +** 1 0.4   3 layer 1-50   
# Model_Path = "./Log/2024-01-27-02-23-52/model/" 

# ## +** 1 0.5   3 layer 1-50   
# Model_Path = "./Log/2024-01-27-02-30-02/model/" 
# # Full_model_45.pkl
# # 1.126185655593872
# # ========== Evaluate Test set ==========
# # Test loss:  0.3367994998892148
# # Test binary acc:  0.8632075471698113
# # Test precision: 0.5546351084812623
# # Test recall:  0.6775903614457831
# # Test f1:  0.6099783080260304
# # Test AUC:  0.8864200573184416
# # Test AUPRC:  0.6518378592842748
# # Test mcc:  0.5319283542121286
# # Threshold:  0.22

# ## +** 1 0.6   3 layer 1-50   
# Model_Path = "./Log/2024-01-27-02-35-34/model/" 
# # Full_model_23.pkl
# # 1.267867088317871
# # ========== Evaluate Test set ==========
# # Test loss:  0.360692643125852
# # Test binary acc:  0.8443396226415094
# # Test precision: 0.5054986727341676
# # Test recall:  0.6424096385542168
# # Test f1:  0.5657894736842104
# # Test AUC:  0.8662318185924655
# # Test AUPRC:  0.5835086352187008
# # Test mcc:  0.47763892129139796
# # Threshold:  0.26

# ## +** 1 0.7   3 layer 1-50   
# Model_Path = "./Log/2024-01-27-02-41-13/model/" 
# # Full_model_38.pkl
# # 1.0446102619171143
# # ========== Evaluate Test set ==========
# # Test loss:  0.4402919997771581
# # Test binary acc:  0.8609251369446135
# # Test precision: 0.5522198731501057
# # Test recall:  0.6293975903614458
# # Test f1:  0.5882882882882883
# # Test AUC:  0.869189650461998
# # Test AUPRC:  0.617795733742503
# # Test mcc:  0.506612585455119
# # Threshold:  0.09

# # ## +** 1    1 layer 1-50   
# # Model_Path = "./Log/2024-01-27-02-49-32/model/" 

# # ## +** 1    2 layer 1-50   
# # Model_Path = "./Log/2024-01-27-02-49-32/model/" 

# Model_Path = "./Log/2024-01-22-19-01-23/model/"
# Model_Path = "/home/bio-3090ti/BenzcodeL/Log/2025-08-17-21-16-26/model/"

# Model_Path = "/home/bio-3090ti/BenzcodeL/Log/2025-08-18-00-38-55/model/"# self.criterion = FocalLoss(gamma=2.0, alpha=0.75)
# Model_Path = "/home/bio-3090ti/BenzcodeL/Log/2025-08-18-04-58-09/model/"# self.criterion = FocalLoss(gamma=2.0, alpha=torch.tensor([0.15, 0.85])
# Model_Path = "/home/bio-3090ti/BenzcodeL/Log/2025-08-18-21-36-16/model/"# tensor([0.1, 0.9],
Model_Path = "/home/bio-3090ti/BenzcodeL/Log/2025-10-05-18-03-18 DE/model/"# FocalLoss(gamma=0.5, alpha=0.85) # Unbalance

print(Model_Path)

time_data = [0,0,0,0]


def save_time_data(time_list,seq_lenth,datapath):
    time_data = {}
    time_data['time_list'] = time_list
    time_data['seq_lenth'] = seq_lenth
    with open(datapath, "wb") as f:
        pickle.dump(time_data, f)



def evaluate(model, data_loader):
    model.eval()

    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []
    pred_dict = {}
    test_dic = {}
    agat_dic = {}
    true_dic = {}
    time_list = []
    seq_lenth = []
    
    for data in data_loader:
        with torch.no_grad():
            sequence_names, seq, labels, node_features, G_batch, adj_matrix, pos = data
            # sequence_names, seq, labels, node_features, G_batch, adj_matrix = data
            # print(seq)
            # exit()
            # print(pos)
            # exit()

            if torch.cuda.is_available():
                node_features = Variable(node_features.cuda().float())
                adj_matrix = Variable(adj_matrix.cuda())
                G_batch.edata['ex'] = Variable(G_batch.edata['ex'].float())
                G_batch = G_batch.to(torch.device('cuda:0'))
                y_true = Variable(labels.cuda())
                pos =Variable(pos.cuda().float())


            else:
                node_features = Variable(node_features.float())
                adj_matrix = Variable(adj_matrix)
                y_true = Variable(labels)
                G_batch.edata['ex'] = Variable(G_batch.edata['ex'].float())

            adj_matrix = torch.squeeze(adj_matrix)
            y_true = torch.squeeze(y_true)
            y_true = y_true.long()

            # y_pred = model(node_features, G_batch, adj_matrix)
            y_pred = model(node_features, G_batch, adj_matrix, pos)
            
            
            # print(y_pred)
            # print(y_true)
            loss = model.criterion(y_pred, y_true)
            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred)
            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()
            valid_pred += [pred[1] for pred in y_pred]
            valid_true += list(y_true)
            pred_dict[sequence_names[0]] = [pred[1] for pred in y_pred]

            # print(sequence_names[0])
            # print(list(y_true))
            # 输出位点
            test_pred = [pred[1] for pred in y_pred]
            # print([1 if pr > 0.18  else 0 for pr in test_pred])
            test_dic[sequence_names[0]] = [1 if pr >= 0.18  else 0 for pr in test_pred]
            true_dic[sequence_names[0]] = list(y_true)

            epoch_loss += loss.item()
            # print(test_dic)
            # print(n)
            n += 1
    
    # save_time_data(time_list,seq_lenth,'./Plot/agat_time.p')
    epoch_loss_avg = epoch_loss / n

    # f_test = "./Sites/test.p"
    # f_true = "./Sites/true.p"
    f_agat = "/home/bio-3090ti/BenzcodeL/MEGPPIS/Sites/agat.p"
    # print(test_dic)
    # with open(f_test, "wb") as f:
    #     pickle.dump(test_dic, f)
    # with open(f_true, "wb") as f:
    #     pickle.dump(true_dic, f)
    # exit()
    with open(f_agat, "wb") as f:
        pickle.dump(test_dic, f)
    
    
    return epoch_loss_avg, valid_true, valid_pred, pred_dict


def plot(y_test,y_score):
    # 计算
    fpr, tpr, thread = metrics.roc_curve(y_test, y_score)
    roc_auc = metrics.auc(fpr, tpr)
    # 绘图
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('AUC.png',)

def save_plot_data(binary_true, y_pred, datapath):

    dic = {}
    dic['y_true'] = binary_true
    dic['y_pred'] = y_pred

    with open(datapath, "wb") as f:
        pickle.dump(dic, f)



def analysis(y_true, y_pred, best_threshold = None):
    if best_threshold == None:
        best_f1 = 0
        best_threshold = 0
        for threshold in range(0, 100):
            threshold = threshold / 100
            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            binary_true = y_true
            f1 = metrics.f1_score(binary_true, binary_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    # plot(binary_true, y_pred)

    
    # save_time_data()
    binary_pred = [1 if pred >= best_threshold else 0 for pred in y_pred]
    binary_true = y_true

    # save_plot_data(binary_true, binary_pred ,datapath='./Plot/re_agat1_best.p')
    # exit()
    
    # binary evaluate
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    AUC = metrics.roc_auc_score(binary_true, y_pred)
    precisions, recalls, thresholds = metrics.precision_recall_curve(binary_true, y_pred)
    AUPRC = metrics.auc(recalls, precisions)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)

    results = {
        'binary_acc': binary_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'AUC': AUC,
        'AUPRC': AUPRC,
        'mcc': mcc,
        'threshold': best_threshold
    }
    return results
def save_csv(model_name, result_test, name):
    # Assuming you have a CSV file path
    dir_name = Model_Path[0:-6] +'result/'

    csv_file_path = Model_Path[0:-6] +'result/' + name + ".csv"

    # 0817 Ben
    import os
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    ###  0817
    
    # Data to be written to the CSV file
    data = [
        ["Test loss", "Test binary acc", "Test precision", "Test recall", "Test f1", "Test AUC", "Test AUPRC", "Test mcc", "Threshold"],
        [model_name, result_test['binary_acc'], result_test['precision'], result_test['recall'],
        result_test['f1'], result_test['AUC'], result_test['AUPRC'], result_test['mcc'], result_test['threshold']]
    ]
    file_exists = os.path.isfile(csv_file_path)

    # Open the CSV file in append mode
    with open(csv_file_path, mode='a', newline='') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)

        # Write the header if the file is new
        if not file_exists:
            csv_writer.writerow(data[0])

        # Write the data to the CSV file
        csv_writer.writerow(data[1])
    

def test(test_dataframe, psepos_path, dataset_name):
    test_loader = DataLoader(dataset=ProDataset(dataframe=test_dataframe,psepos_path=psepos_path), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=graph_collate)

    for model_name in sorted(os.listdir(Model_Path)):
        print(model_name)
        
        model = AGATPPIS(LAYER, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT, LAMBDA, ALPHA)
        ## ben 0817
        # print(model)
        ##
        if torch.cuda.is_available():
            model.cuda()
        model.load_state_dict(torch.load(Model_Path + model_name, map_location='cuda:0'))
        start_time = time.time()  # 记录开始时间
        epoch_loss_test_avg, test_true, test_pred, pred_dict = evaluate(model, test_loader)
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算耗时
        print(elapsed_time)

        if dataset_name == 'Test_60':
            time_data[0] += elapsed_time
        elif dataset_name == 'Test_315-28':
            time_data[1] += elapsed_time
        elif dataset_name == 'Btest_31-6':
            time_data[2] += elapsed_time
        else:
            time_data[3] += elapsed_time
        # print(pred_dict)
        # exit()

        result_test = analysis(test_true, test_pred)
        save_csv(model_name,result_test,dataset_name)

        
        print("========== Evaluate Test set ==========")
        print("Test loss: ", epoch_loss_test_avg)
        print("Test binary acc: ", result_test['binary_acc'])
        print("Test precision:", result_test['precision'])
        print("Test recall: ", result_test['recall'])
        print("Test f1: ", result_test['f1'])
        print("Test AUC: ", result_test['AUC'])
        print("Test AUPRC: ", result_test['AUPRC'])
        print("Test mcc: ", result_test['mcc'])
        print("Threshold: ", result_test['threshold'])
        

def test_one_dataset(dataset, psepos_path, dataset_name):
    IDs, sequences, labels = [], [], []
    for ID in dataset:
        IDs.append(ID)
        item = dataset[ID]
        sequences.append(item[0])
        labels.append(item[1])
    test_dic = {"ID": IDs, "sequence": sequences, "label": labels}
    test_dataframe = pd.DataFrame(test_dic)
    test(test_dataframe, psepos_path, dataset_name)


def main():
    with open(Dataset_Path + "Test_60.pkl", "rb") as f:
        Test_60 = pickle.load(f)

    with open(Dataset_Path + "Test_315-28.pkl", "rb") as f:
        Test_315_28 = pickle.load(f)

    with open(Dataset_Path + "UBtest_31-6.pkl", "rb") as f:
        UBtest_31_6 = pickle.load(f)

    Btest_31_6 = {}
    with open(Dataset_Path + "bound_unbound_mapping31-6.txt", "r") as f:
        lines = f.readlines()[1:]
    for line in lines:
        bound_ID, unbound_ID, _ = line.strip().split()
        Btest_31_6[bound_ID] = Test_60[bound_ID]

    Test60_psepos_Path = '/home/bio-3090ti/BenzcodeL/MEGPPIS/Feature/psepos/Test60_psepos_SC.pkl'
    Test315_28_psepos_Path = '/home/bio-3090ti/BenzcodeL/MEGPPIS/Feature/psepos/Test315-28_psepos_SC.pkl'
    Btest31_psepos_Path = '/home/bio-3090ti/BenzcodeL/MEGPPIS/Feature/psepos/Test60_psepos_SC.pkl'
    UBtest31_28_psepos_Path = '/home/bio-3090ti/BenzcodeL/MEGPPIS/Feature/psepos/UBtest31-6_psepos_SC.pkl'

    # Test60_psepos_Path = './Feature/psepos/Test60_psepos_C.pkl'
    # Test315_28_psepos_Path = './Feature/psepos/Test315-28_psepos_C.pkl'
    # Btest31_psepos_Path = './Feature/psepos/Test60_psepos_C.pkl'
    # UBtest31_28_psepos_Path = './Feature/psepos/UBtest31-6_psepos_C.pkl'

    print("Evaluate GraphPPIS on Test_60")
    test_one_dataset(Test_60, Test60_psepos_Path, 'Test_60')
    # exit()
    print("Evaluate GraphPPIS on Test_315-28")
    test_one_dataset(Test_315_28, Test315_28_psepos_Path,'Test_315-28')

    # print("Evaluate GraphPPIS on Btest_31-6")
    # test_one_dataset(Btest_31_6, Btest31_psepos_Path, 'Btest_31-6')

    # print("Evaluate GraphPPIS on UBtest_31-6")
    # test_one_dataset(UBtest_31_6, UBtest31_28_psepos_Path, 'UBtest_31-6')
    # for i in range(5):
    #     print("Evaluate GraphPPIS on Test_60")
    #     test_one_dataset(Test_60, Test60_psepos_Path, 'Test_60')

    #     print("Evaluate GraphPPIS on Test_315-28")
    #     test_one_dataset(Test_315_28, Test315_28_psepos_Path,'Test_315-28')

    #     print("Evaluate GraphPPIS on Btest_31-6")
    #     test_one_dataset(Btest_31_6, Btest31_psepos_Path, 'Btest_31-6')

    #     print("Evaluate GraphPPIS on UBtest_31-6")
    #     test_one_dataset(UBtest_31_6, UBtest31_28_psepos_Path, 'UBtest_31-6')
    
    # for ti in time_data:
    #     print(ti/5.0)

# 1.1845252990722657
# 3.640677642822266
# 0.6809999465942382
# 0.7066010951995849
        
# 6.223786735534668
# 26.544369411468505
# 2.9441261291503906
# 2.9462968826293947

def save_time_results():
    """保存时间结果到文件"""
    with open('/home/bio-3090ti/BenzcodeL/Log/2025-08-25-17-33-34 BESTT/meg_runtime_results.txt', 'a') as f:
        f.write(f"Run at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test_60: {time_data[0]:.2f}s\n")
        f.write(f"Test_315-28: {time_data[1]:.2f}s\n")
        f.write(f"Btest_31-6: {time_data[2]:.2f}s\n")
        f.write(f"UBtest_31-6: {time_data[3]:.2f}s\n")
        f.write("="*50 + "\n")

if __name__ == "__main__":
    main()
    save_time_results()
