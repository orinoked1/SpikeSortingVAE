import pandas as pd
import numpy as np
from pathlib import Path


def createDict(*args):
    return dict(((k, eval(k)) for k in args))

summery = pd.DataFrame()
gt_files = list(Path(r"D:\Drive\DL_project_OD\reports").rglob("*fet*.csv"))
for file in gt_files:
    data = pd.read_csv(str(file))
    data = data[data.good_class==1]
    if file.name.endswith("all.csv"):
        if file.name.find("fet.1")>0:
            train_all_gt = data
            LD = 44
        else:
            test_all_gt = data
            LD = 36
    else:
        if file.name.find("fet.1")>0:
            train_gt = data
            LD = 44
        else:
            test_gt = data
            LD = 36
    if file.name.find("_C")>0:
        C = file.name[file.name.find("_C")+2:]
        LD = float(C[:1])*4
    mean_id = np.mean(data.ID)
    sd_id = np.std(data.ID)
    mean_l_ratio = np.mean(data.L_ratio)
    sd_l_ratio = np.std(data.L_ratio)
    chs = np.mean(data.CHS)
    dbs = np.mean(data.DBS)
    mean_id_change = 1
    sd_id_change = 1
    mean_l_ratio_change = 1
    sd_l_ratio_change = 1
    chs_change = 1
    dbs_change = 1
    DO=-1
    LR=-1
    include_0 = file.name.endswith("all.csv")>0
    stage=-1
    architecture = -1
    train = file.name.find("fet.1")>0
    name = file.name
    F = -1
    C = -1
    ## Capping
    f_capp = data[['L_ratio', 'ID']]
    f_capp.loc[f_capp['L_ratio'] < 0.1, 'L_ratio'] = 0.1
    f_capp.loc[f_capp['ID'] > 50, 'ID'] = 50
    mean_capp_id = np.mean(f_capp.ID)
    sd_capp_id = np.std(f_capp.ID)
    mean_capp_lr = np.mean(f_capp.L_ratio)
    sd_capp_lr = np.std(f_capp.L_ratio)

    ## Count good clusters
    num_good_clusters_ID = sum(f_capp['ID'] == 50)
    num_good_clusters_LR = sum(f_capp['L_ratio'] == 0.1)


    dict_var = createDict('name','train','stage','include_0','architecture','LD','LR','DO','F','C',
                          'mean_id','mean_id_change','sd_id_change',
                          'mean_l_ratio','sd_l_ratio','mean_l_ratio_change','sd_l_ratio_change',
                          'chs','chs_change','dbs','dbs_change', 'mean_capp_id', 'sd_capp_id', 
                          'mean_capp_lr', 'sd_capp_lr', 'num_good_clusters_ID', 'num_good_clusters_LR')
    cur_df = pd.DataFrame(dict_var, index=[0])
    summery =summery.append(cur_df)

file_list = list(Path(r"D:\Drive\DL_project_OD\reports").rglob("*vae*.csv"))
for file in file_list:
    if file.name.startswith("vae"):
        channels_dependent = True
        S = file.name[file.name.find("Stage")+5:]
        stage = int(S[:1])
    else:
        channels_dependent = False
        S = file.name[file.name.find("stage_")+6:]
        stage = int(S[:1])
    LD = file.name[file.name.find("_LD")+3:]
    LD = int(LD[:LD.find("_")])
    DO = file.name[file.name.find("_DR")+3:]
    if DO.find("_")>0:
        DO = float(DO[:DO.find("_")])
    else:
        DO = float(DO[:DO.find(".pt")])
    if file.name.find("_F")>0:
        F = file.name[file.name.find("_F")+2:]
        F = float(F[:F.find(".pt")])
    else:
        F=0
    if file.name.find("_C")>0:
        C = file.name[file.name.find("_C")+2:]
        C = float(C[:1])
    else:
        C=9
    LR = file.name[file.name.find("_LR")+3:]
    LR = float(LR[:LR.find("_")])
    if file.name.endswith("all.csv"):
        include_0 = True
    else:
        include_0 = False
    if file.name.startswith("simple_vae_stage_1"):
        architecture = 1
    elif file.name.startswith("resnet_vae_stage_"):
        architecture = 2
    elif file.name.startswith("resnet_vaeV2_stage"):
        architecture = 3
    else:
        architecture = 0
    if file.name.find("Train") > 0:
        train = True
    else:
        train = False
    data = pd.read_csv(str(file))
    data = data[data.good_class==1]

    if include_0:
        if train:
            curr_gt = train_all_gt
        else:
            curr_gt = test_all_gt
    else:
        if train:
            curr_gt = train_gt
        else:
            curr_gt = test_gt
    
    ## Capping
    f_capp = data[['L_ratio', 'ID']]
    f_capp.loc[f_capp['L_ratio'] < 0.1, 'L_ratio'] = 0.1
    f_capp.loc[f_capp['ID'] > 50, 'ID'] = 50
    mean_capp_id = np.mean(f_capp.ID)
    sd_capp_id = np.std(f_capp.ID)
    mean_capp_lr = np.mean(f_capp.L_ratio)
    sd_capp_lr = np.std(f_capp.L_ratio)

    ## Count good clusters
    num_good_clusters_ID = sum(f_capp['ID'] == 50)
    num_good_clusters_LR = sum(f_capp['L_ratio'] == 0.1)


    mean_id = np.mean(data.ID)
    sd_id = np.std(data.ID)
    mean_l_ratio = np.mean(data.L_ratio)
    sd_l_ratio = np.std(data.L_ratio)
    if 'CHS' in data.columns:
        chs = np.mean(data.CHS)
        chs_change = np.mean(data.CHS) / np.mean(curr_gt.CHS)
    else:
        chs = -1
        chs_change=-1
    if 'DBS' in data.columns:
        dbs = np.mean(data.DBS)
        dbs_change = np.mean(data.DBS) / np.mean(curr_gt.DBS)
    else:
        dbs = -1
        dbs_change=-1
    mean_id_change = np.mean(data.ID/curr_gt.ID)
    sd_id_change = np.std(data.ID/curr_gt.ID)
    mean_l_ratio_change = np.mean(data.L_ratio/curr_gt.L_ratio)
    sd_l_ratio_change = np.std(data.L_ratio/curr_gt.L_ratio)
    name = file.name


    dict_var = createDict('name','train','stage','include_0','architecture','LD','LR','DO','F','C',
                          'mean_id','mean_id_change','sd_id_change',
                          'mean_l_ratio','sd_l_ratio','mean_l_ratio_change','sd_l_ratio_change',
                          'chs','chs_change','dbs','dbs_change', 'mean_capp_id', 'sd_capp_id', 
                          'mean_capp_lr', 'sd_capp_lr', 'num_good_clusters_ID', 'num_good_clusters_LR')
    cur_df = pd.DataFrame(dict_var, index=[0])
    summery =summery.append(cur_df)
summery.to_csv('summery.csv')
