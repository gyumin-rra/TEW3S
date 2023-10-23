import copy
import pandas as pd
import numpy as np
import datetime

def data_concatenation(hadmid, filling_dict, data_sources_dict, predictor_df):
    item_list = predictor_df['items'].tolist()
    tmp_dict = copy.deepcopy(filling_dict)
    tmp_dict['hospadm_id'] = hadmid

    # age
    icustays = data_sources_dict['icustays']
    tmp_cond = icustays.hadm_id == hadmid
    tmp_data = icustays.loc[tmp_cond].reset_index(drop = True)
    tmp_IT = tmp_data.intime[0]
    tmp_dict['age']['charttime'] = tmp_IT
    tmp_dict['age']['value'] = tmp_data.age_at_intime[0]

    # gender
    tmp_dict['gender']['value'] = tmp_data.gender[0]

    # race
    tmp_dict['race']['value'] = tmp_data.race[0]

    vaso_list = ['epinephrine', 'dopamine', 'dobutamine', 'norepinephrine', 'phenylephrine', 'vasopressin']

    for item in item_list:
        tmp_cond = predictor_df['items'] == item
        df_names = predictor_df.iloc[tmp_cond[tmp_cond].index, 1:].T.dropna().index.tolist()
        itemid_list = predictor_df.iloc[tmp_cond[tmp_cond].index, 1:].dropna(axis=1).to_numpy()[0].tolist()

        for idx, dn in enumerate(df_names):
            data = data_sources_dict[dn]
            if dn == 'omr':
                tmp_itemid = [itemid_list[idx]]
            else:
                tmp_itemid = np.array(itemid_list[idx].split(',')).astype(int).tolist() if itemid_list[idx].__contains__(',') else [np.array(itemid_list[idx]).astype(int).tolist()]
            
            tmp_cond = (data.hadm_id == hadmid) & (data.itemid.isin(tmp_itemid))

            if sum(tmp_cond) == 0:
                continue

            if item in vaso_list:
                tmp_data = data.loc[tmp_cond]

                tmp_dict[item]['starttime'] = tmp_data.starttime.tolist()
                tmp_dict[item]['endtime'] = tmp_data.endtime.tolist()
                tmp_dict[item]['value'] = tmp_data.amount.tolist()
                tmp_dict[item]['rate'] = tmp_data.rate.tolist()
                
            elif item in ['fluid']:
                tmp_data = data.loc[tmp_cond]

                tmp_dict[item]['starttime'] = tmp_data.starttime.tolist()
                tmp_dict[item]['endtime'] = tmp_data.endtime.tolist()
                tmp_dict[item]['value'] = tmp_data.amount.tolist()

            elif item in ['ventilator']:
                tmp_data = data.loc[tmp_cond]

                tmp_dict[item]['starttime'] = tmp_data.starttime.tolist()
                tmp_dict[item]['endtime'] = tmp_data.endtime.tolist()
                tmp_dict[item]['value'] = np.where(tmp_data.value > 0, 1, 0).tolist()

            elif item in ['weight']:
                tmp_data = data.loc[tmp_cond]

                tmp_dict[item]['charttime'] = tmp_dict[item]['charttime'] + tmp_data.charttime.tolist()
                tmp_dict[item]['value'] = tmp_dict[item]['value'] + tmp_data.value.tolist()

                if idx == 0:
                    data = data_sources_dict['IE']
                    tmp_cond = (data.hadm_id == hadmid) & (~pd.isna(data.patientweight))
                    tmp_data = data.loc[tmp_cond] 

                    tmp_dict[item]['charttime'] = tmp_dict[item]['charttime'] + tmp_data.starttime.tolist()
                    tmp_dict[item]['value'] = tmp_dict[item]['value'] + tmp_data.patientweight.tolist()
                
            else:
                tmp_data = data.loc[tmp_cond]

                tmp_dict[item]['charttime'] = tmp_dict[item]['charttime'] + tmp_data.charttime.tolist()
                tmp_dict[item]['value'] = tmp_dict[item]['value'] + tmp_data.value.tolist()
    
    for item in item_list:
        if item in vaso_list:
            tmp_data = pd.DataFrame({
                'starttime' : tmp_dict[item]['starttime'],
                'endtime' : tmp_dict[item]['endtime'],
                'value': tmp_dict[item]['value'],
                'rate': tmp_dict[item]['rate']
            })

            if tmp_data.shape[0] == 0:
                continue
            
            tmp_data = tmp_data.sort_values('starttime').reset_index(drop=True)

            tmp_dict[item]['starttime'] = tmp_data.starttime.tolist()
            tmp_dict[item]['endtime'] = tmp_data.endtime.tolist()
            tmp_dict[item]['value'] = tmp_data.value.tolist()
            tmp_dict[item]['rate'] = tmp_data.rate.tolist()

        elif item in ['fluid', 'ventilator']:
            tmp_data = pd.DataFrame({
                'starttime' : tmp_dict[item]['starttime'],
                'endtime' : tmp_dict[item]['endtime'],
                'value': tmp_dict[item]['value']
            })

            if tmp_data.shape[0] == 0:
                continue
            
            tmp_data = tmp_data.sort_values('starttime').reset_index(drop=True)

            tmp_dict[item]['starttime'] = tmp_data.starttime.tolist()
            tmp_dict[item]['endtime'] = tmp_data.endtime.tolist()
            tmp_dict[item]['value'] = tmp_data.value.tolist()

        else: 
            tmp_data = pd.DataFrame({
                'charttime' : tmp_dict[item]['charttime'],
                'value': tmp_dict[item]['value']
            })

            if tmp_data.shape[0] == 0:
                continue
            
            tmp_data = tmp_data.sort_values('charttime').reset_index(drop=True)

            tmp_dict[item]['charttime'] = tmp_data.charttime.tolist()
            tmp_dict[item]['value'] = tmp_data.value.tolist()
    

    # derived varibles: MAP, BMI======================================================================================================================================================================
    tmp_data = pd.DataFrame({
        'charttime' : tmp_dict['sbp']['charttime'] + tmp_dict['dbp']['charttime'],
        'sbp': tmp_dict['sbp']['value'] + list(np.repeat(np.nan, len(tmp_dict['dbp']['value']))),
        'dbp': list(np.repeat(np.nan, len(tmp_dict['sbp']['value']))) + tmp_dict['dbp']['value']
    })
    tmp_data = tmp_data.sort_values('charttime').reset_index(drop=True)
    tmp_data = tmp_data.ffill()
    tmp_data = tmp_data.bfill()
    tmp_data['map'] = (tmp_data.sbp + 2*tmp_data.dbp)/3
    tmp_dict['map']['charttime'] = tmp_data['charttime'].tolist()
    tmp_dict['map']['value'] = tmp_data['map'].tolist()

    tmp_dict['sbp']['charttime'] = tmp_data['charttime'].tolist()
    tmp_dict['sbp']['value'] = tmp_data['sbp'].tolist()

    tmp_dict['dbp']['charttime'] = tmp_data['charttime'].tolist()
    tmp_dict['dbp']['value'] = tmp_data['dbp'].tolist()


    tmp_data = pd.DataFrame({
        'charttime' : tmp_dict['height']['charttime'] + tmp_dict['weight']['charttime'],
        'height': tmp_dict['height']['value'] + list(np.repeat(np.nan, len(tmp_dict['weight']['value']))),
        'weight': list(np.repeat(np.nan, len(tmp_dict['height']['value']))) + tmp_dict['weight']['value']
    })

    if tmp_data.shape[0] > 0:
        tmp_data = tmp_data.sort_values('charttime').reset_index(drop=True)
        tmp_data = tmp_data.set_index('charttime').interpolate(method='time').reset_index()
        tmp_data = tmp_data.bfill()

        tmp_data['bmi'] = (tmp_data.weight)/(tmp_data.height*tmp_data.height)*10000
        tmp_dict['bmi']['charttime'] = tmp_data['charttime'].tolist()
        tmp_dict['bmi']['value'] = tmp_data['bmi'].tolist()

        tmp_dict['height']['charttime'] = tmp_data['charttime'].tolist()
        tmp_dict['height']['value'] = tmp_data['height'].tolist()

        tmp_dict['weight']['charttime'] = tmp_data['charttime'].tolist()
        tmp_dict['weight']['value'] = tmp_data['weight'].tolist()


    # labeling===========================================================================================================================================================================================
    item_list = list(tmp_dict.keys())
    item_list = item_list[4:]

    # building timeline
    timeline = []
    for i in item_list:
        try:
            timeline = timeline + tmp_dict[i]['charttime']
        except:
            timeline = timeline + tmp_dict[i]['starttime'] + tmp_dict[i]['endtime']
    timeline = pd.Series(timeline).unique()
    timeline = pd.to_datetime(timeline).sort_values()

    tmp_data = pd.DataFrame({
        'starttime': timeline[0:len(timeline)-1],
        'endtime': timeline[1:]
    })

    
    # CNS_SOFA
    tmp_data['gcs'] = np.repeat(np.nan, tmp_data.shape[0])
    tmp_gcs = tmp_dict['gcs']['value']
    tmp_CT = tmp_dict['gcs']['charttime']
    for i in range(len(tmp_CT)):
        tmp_cond = tmp_data.starttime == tmp_CT[i]
        tmp_data.loc[tmp_cond, 'gcs'] = tmp_gcs[i]
    tmp_data = tmp_data.fillna(method='ffill')
    tmp_data = tmp_data.fillna(method='bfill')
    tmp_sofa = np.repeat(0, tmp_data.shape[0])
    tmp_gcs = tmp_data.gcs.tolist()

    for i in range(len(tmp_gcs)):
        if tmp_gcs[i] < 6:
            tmp_sofa[i] = 4
        elif (tmp_gcs[i] >= 6) & (tmp_gcs[i] <= 9):
            tmp_sofa[i] = 3
        elif (tmp_gcs[i] >= 10) & (tmp_gcs[i] <= 12):
            tmp_sofa[i] = 2
        elif (tmp_gcs[i] >= 13) & (tmp_gcs[i] <= 14):
            tmp_sofa[i] = 1
    
    tmp_data['CNS_SOFA'] = tmp_sofa
    

    # CARDIO_SOFA
    tmp_data['map'] = np.repeat(np.nan, tmp_data.shape[0])
    tmp_data['epi_rate'] = np.repeat(0, tmp_data.shape[0])
    tmp_data['dop_rate'] = np.repeat(0, tmp_data.shape[0])
    tmp_data['dob_rate'] = np.repeat(0, tmp_data.shape[0])
    tmp_data['nor_rate'] = np.repeat(0, tmp_data.shape[0])
    tmp_data['phe_rate'] = np.repeat(0, tmp_data.shape[0])#for shock
    tmp_data['vas_rate'] = np.repeat(0, tmp_data.shape[0])#for shock

    tmp_MAP = tmp_dict['map']['value']
    tmp_CT = tmp_dict['map']['charttime']
    for i in range(len(tmp_CT)):
        tmp_cond = tmp_data.starttime == tmp_CT[i]
        tmp_data.loc[tmp_cond, 'map'] = tmp_MAP[i]
    
    tmp_epi_rate = tmp_dict['epinephrine']['rate']
    tmp_ST = tmp_dict['epinephrine']['starttime']
    tmp_ET = tmp_dict['epinephrine']['endtime']
    for i in range(len(tmp_ST)):
        tmp_cond = (tmp_data.starttime >= tmp_ST[i]) & (tmp_data.endtime <= tmp_ET[i]) & (tmp_data.epi_rate < tmp_epi_rate[i])
        tmp_data.loc[tmp_cond, 'epi_rate'] = tmp_epi_rate[i]
    
    tmp_dop_rate = tmp_dict['dopamine']['rate']
    tmp_ST = tmp_dict['dopamine']['starttime']
    tmp_ET = tmp_dict['dopamine']['endtime']
    for i in range(len(tmp_ST)):
        tmp_cond = (tmp_data.starttime >= tmp_ST[i]) & (tmp_data.endtime <= tmp_ET[i]) & (tmp_data.dop_rate < tmp_dop_rate[i])
        tmp_data.loc[tmp_cond, 'dop_rate'] = tmp_dop_rate[i]
    
    tmp_dob_rate = tmp_dict['dobutamine']['rate']
    tmp_ST = tmp_dict['dobutamine']['starttime']
    tmp_ET = tmp_dict['dobutamine']['endtime']
    for i in range(len(tmp_ST)):
        tmp_cond = (tmp_data.starttime >= tmp_ST[i]) & (tmp_data.endtime <= tmp_ET[i]) & (tmp_data.dob_rate < tmp_dob_rate[i])
        tmp_data.loc[tmp_cond, 'dob_rate'] = tmp_dob_rate[i]
    
    tmp_nor_rate = tmp_dict['norepinephrine']['rate']
    tmp_ST = tmp_dict['norepinephrine']['starttime']
    tmp_ET = tmp_dict['norepinephrine']['endtime']
    for i in range(len(tmp_ST)):
        tmp_cond = (tmp_data.starttime >= tmp_ST[i]) & (tmp_data.endtime <= tmp_ET[i]) & (tmp_data.nor_rate < tmp_nor_rate[i])
        tmp_data.loc[tmp_cond, 'nor_rate'] = tmp_nor_rate[i]

    tmp_phe_rate = tmp_dict['phenylephrine']['rate']
    tmp_ST = tmp_dict['phenylephrine']['starttime']
    tmp_ET = tmp_dict['phenylephrine']['endtime']
    for i in range(len(tmp_ST)):
        tmp_cond = (tmp_data.starttime >= tmp_ST[i]) & (tmp_data.endtime <= tmp_ET[i]) & (tmp_data.phe_rate < tmp_phe_rate[i])
        tmp_data.loc[tmp_cond, 'phe_rate'] = tmp_phe_rate[i]
    
    tmp_vas_rate = tmp_dict['vasopressin']['rate']
    tmp_ST = tmp_dict['vasopressin']['starttime']
    tmp_ET = tmp_dict['vasopressin']['endtime']
    for i in range(len(tmp_ST)):
        tmp_cond = (tmp_data.starttime >= tmp_ST[i]) & (tmp_data.endtime <= tmp_ET[i]) & (tmp_data.vas_rate < tmp_vas_rate[i])
        tmp_data.loc[tmp_cond, 'vas_rate'] = tmp_vas_rate[i]
    
    tmp_data = tmp_data.fillna(method='ffill')# filling map
    tmp_data = tmp_data.fillna(method='bfill')# filling map

    tmp_sofa = np.repeat(0, tmp_data.shape[0])
    tmp_MAP = tmp_data['map'].tolist()
    tmp_epi_rate = tmp_data['epi_rate'].tolist() 
    tmp_dop_rate = tmp_data['dop_rate'].tolist()
    tmp_dob_rate = tmp_data['dob_rate'].tolist()
    tmp_nor_rate = tmp_data['nor_rate'].tolist()
    tmp_phe_rate = tmp_data['phe_rate'].tolist()
    tmp_vas_rate = tmp_data['vas_rate'].tolist()

    for i in range(len(tmp_sofa)):
        if tmp_MAP[i] < 70:
            tmp_sofa[i] = 1
            if (tmp_dop_rate[i] > 0) | (tmp_dob_rate[i] > 0):
                tmp_sofa[i] = 2
            if (tmp_dop_rate[i] > 5) | ((tmp_epi_rate[i] > 0) & (tmp_epi_rate[i] <= 0.1))| ((tmp_nor_rate[i] > 0) & (tmp_nor_rate[i] <= 0.1)):
                tmp_sofa[i] = 3
            if (tmp_dop_rate[i] > 15) | (tmp_epi_rate[i] > 0.1) | (tmp_nor_rate[i] > 0.1):
                tmp_sofa[i] = 4
            
    tmp_data['CARDIO_SOFA'] = tmp_sofa
    tmp_cond = (tmp_data['epi_rate']>0) | (tmp_data['dop_rate']>0) | (tmp_data['dob_rate']>0) | (tmp_data['nor_rate']>0) | (tmp_data['phe_rate']>0) | (tmp_data['vas_rate']>0)
    tmp = np.where(tmp_cond, 1, 0).tolist()
    tmp_data['vaso_presence'] = tmp
    

    # RESP_SOFA
    tmp_data['pao2'] = np.repeat(np.nan, tmp_data.shape[0])
    tmp_data['ventilator'] = np.repeat(0, tmp_data.shape[0])
    tmp_pao2 = tmp_dict['pao2']['value']
    tmp_fio2 = tmp_dict['fio2']['value']
    if len(tmp_fio2)==0:
        tmp_data['fio2'] = np.repeat(21, tmp_data.shape[0])
    else:
        tmp_data['fio2'] = np.repeat(np.nan, tmp_data.shape[0])
    
    tmp_vent = tmp_dict['ventilator']['value']

    tmp_CT = tmp_dict['pao2']['charttime']
    for i in range(len(tmp_CT)):
        tmp_cond = tmp_data.starttime == tmp_CT[i]
        tmp_data.loc[tmp_cond, 'pao2'] = tmp_pao2[i]
    
    tmp_CT = tmp_dict['fio2']['charttime']
    for i in range(len(tmp_CT)):
        tmp_cond = tmp_data.starttime == tmp_CT[i]
        tmp_data.loc[tmp_cond, 'fio2'] = tmp_fio2[i]

    tmp_ST = tmp_dict['ventilator']['starttime']
    tmp_ET = tmp_dict['ventilator']['endtime']
    for i in range(len(tmp_ST)):
        tmp_cond = (tmp_data.starttime >= tmp_ST[i]) & (tmp_data.endtime <= tmp_ET[i]) & (tmp_data.loc[tmp_cond, 'ventilator'] < tmp_vent[i])
        tmp_data.loc[tmp_cond, 'ventilator'] = tmp_vent[i]
    
    tmp_data = tmp_data.fillna(method='ffill')
    tmp_data = tmp_data.fillna(method='bfill')

    tmp_sofa = np.repeat(0, tmp_data.shape[0])
    tmp_pao2 = tmp_data['pao2'].tolist()
    tmp_fio2 = tmp_data['fio2'].tolist()
    tmp_vent = tmp_data['ventilator'].tolist()

    for i in range(len(tmp_sofa)):
        if tmp_vent[i] == 0:
            if ((tmp_pao2[i]/tmp_fio2[i]) < 400) & ((tmp_pao2[i]/tmp_fio2[i]) >= 300):
                tmp_sofa[i] = 1
            elif ((tmp_pao2[i]/tmp_fio2[i]) < 300) & ((tmp_pao2[i]/tmp_fio2[i]) >= 200):
                tmp_sofa[i] = 2
        else:
            if ((tmp_pao2[i]/tmp_fio2[i]) < 200) & ((tmp_pao2[i]/tmp_fio2[i]) >= 100):
                tmp_sofa[i] = 3
            elif ((tmp_pao2[i]/tmp_fio2[i]) < 100) & ((tmp_pao2[i]/tmp_fio2[i]) >= 0):
                tmp_sofa[i] = 4

    tmp_data['RESP_SOFA'] = tmp_sofa


    # COAG_SOFA
    tmp_data['platelets'] = np.repeat(np.nan, tmp_data.shape[0])
    tmp_plat = tmp_dict['platelets']['value']
    tmp_CT = tmp_dict['platelets']['charttime']
    for i in range(len(tmp_CT)):
        tmp_cond = tmp_data.starttime == tmp_CT[i]
        tmp_data.loc[tmp_cond, 'platelets'] = tmp_plat[i]
    tmp_data = tmp_data.fillna(method='ffill')
    tmp_data = tmp_data.fillna(method='bfill')
    tmp_sofa = np.repeat(0, tmp_data.shape[0])
    tmp_plat = tmp_data.platelets.tolist()

    for i in range(len(tmp_plat)):
        if (tmp_plat[i] >= 100) & (tmp_plat[i] < 150):
            tmp_sofa[i] = 1
        elif (tmp_plat[i] >= 50) & (tmp_plat[i] < 100):
            tmp_sofa[i] = 2
        elif (tmp_plat[i] >= 20) & (tmp_plat[i] < 50):
            tmp_sofa[i] = 3
        elif (tmp_plat[i] < 20):
            tmp_sofa[i] = 4
    
    tmp_data['COAG_SOFA'] = tmp_sofa


    # LIVER_SOFA
    tmp_data['bilirubin'] = np.repeat(np.nan, tmp_data.shape[0])
    tmp_bili = tmp_dict['bilirubin']['value']
    tmp_CT = tmp_dict['bilirubin']['charttime']
    for i in range(len(tmp_CT)):
        tmp_cond = tmp_data.starttime == tmp_CT[i]
        tmp_data.loc[tmp_cond, 'bilirubin'] = tmp_bili[i]
    tmp_data = tmp_data.fillna(method='ffill')
    tmp_data = tmp_data.fillna(method='bfill')
    tmp_sofa = np.repeat(0, tmp_data.shape[0])
    tmp_bili = tmp_data.bilirubin.tolist()

    for i in range(len(tmp_bili)):
        if (tmp_bili[i] >= 1.2) & (tmp_bili[i] < 2):
            tmp_sofa[i] = 1
        elif (tmp_bili[i] >= 2) & (tmp_bili[i] < 6):
            tmp_sofa[i] = 2
        elif (tmp_bili[i] >= 6) & (tmp_bili[i] < 12):
            tmp_sofa[i] = 3
        elif (tmp_bili[i] >= 12):
            tmp_sofa[i] = 4
    
    tmp_data['LIVER_SOFA'] = tmp_sofa


    # RENAL_SOFA
    tmp_data['creatinine'] = np.repeat(np.nan, tmp_data.shape[0])
    tmp_creat = tmp_dict['creatinine']['value']
    tmp_CT = tmp_dict['creatinine']['charttime']
    for i in range(len(tmp_CT)):
        tmp_cond = tmp_data.starttime == tmp_CT[i]
        tmp_data.loc[tmp_cond, 'creatinine'] = tmp_creat[i]
    tmp_data = tmp_data.fillna(method='ffill')
    tmp_data = tmp_data.fillna(method='bfill')
    tmp_sofa = np.repeat(0, tmp_data.shape[0])
    tmp_creat = tmp_data.creatinine.tolist()

    for i in range(len(tmp_creat)):
        if (tmp_creat[i] >= 1.2) & (tmp_creat[i] < 2):
            tmp_sofa[i] = 1
        elif (tmp_creat[i] >= 2) & (tmp_creat[i] < 3.5):
            tmp_sofa[i] = 2
        elif (tmp_creat[i] >= 3.5) & (tmp_creat[i] < 5):
            tmp_sofa[i] = 3
        elif (tmp_creat[i] >= 5):
            tmp_sofa[i] = 4
    
    tmp_data['RENAL_SOFA'] = tmp_sofa

    # SOFA
    tmp_data['SOFA'] = tmp_data['CNS_SOFA'] + tmp_data['CARDIO_SOFA'] + tmp_data['RESP_SOFA'] + tmp_data['COAG_SOFA'] + tmp_data['LIVER_SOFA'] + tmp_data['RENAL_SOFA']

    # SEPSIS
    tmp_data['SEPSIS'] = np.where(tmp_data['SOFA']>=2, 1, 0)

    # SEPTIC SHOCK
    tmp_data['lactate'] = np.repeat(np.nan, tmp_data.shape[0])
    tmp_lact = tmp_dict['lactate']['value']
    tmp_CT = tmp_dict['lactate']['charttime']
    for i in range(len(tmp_CT)):
        tmp_cond = tmp_data.starttime == tmp_CT[i]
        tmp_data.loc[tmp_cond, 'lactate'] = tmp_lact[i]
    tmp_data = tmp_data.fillna(method='ffill')
    tmp_data = tmp_data.fillna(method='bfill')
    
    tmp_data['SHOCK'] = np.where((tmp_data['lactate']>=2)&(tmp_data['vaso_presence']==1), 1, 0)

    # input SOFAs, sepsis, shock
    tmp_dict['fio2'] = {
        'charttime' : tmp_data.starttime.tolist(),
        'value' : tmp_data.fio2.tolist()
    }
    tmp_dict['CNS_SOFA'] = {
        'starttime' : tmp_data.starttime.tolist(),
        'endtime' : tmp_data.endtime.tolist(),
        'value' : tmp_data.CNS_SOFA.tolist()
    }
    tmp_dict['CARDIO_SOFA'] = {
        'starttime' : tmp_data.starttime.tolist(),
        'endtime' : tmp_data.endtime.tolist(),
        'value' : tmp_data.CARDIO_SOFA.tolist()
    }
    tmp_dict['RESP_SOFA'] = {
        'starttime' : tmp_data.starttime.tolist(),
        'endtime' : tmp_data.endtime.tolist(),
        'value' : tmp_data.RESP_SOFA.tolist()
    }
    tmp_dict['COAG_SOFA'] = {
        'starttime' : tmp_data.starttime.tolist(),
        'endtime' : tmp_data.endtime.tolist(),
        'value' : tmp_data.COAG_SOFA.tolist()
    }
    tmp_dict['LIVER_SOFA'] = {
        'starttime' : tmp_data.starttime.tolist(),
        'endtime' : tmp_data.endtime.tolist(),
        'value' : tmp_data.LIVER_SOFA.tolist()
    }
    tmp_dict['RENAL_SOFA'] = {
        'starttime' : tmp_data.starttime.tolist(),
        'endtime' : tmp_data.endtime.tolist(),
        'value' : tmp_data.RENAL_SOFA.tolist()
    }
    tmp_dict['SOFA'] = {
        'starttime' : tmp_data.starttime.tolist(),
        'endtime' : tmp_data.endtime.tolist(),
        'value' : tmp_data.SOFA.tolist()
    }
    tmp_dict['SEPSIS'] = {
        'starttime' : tmp_data.starttime.tolist(),
        'endtime' : tmp_data.endtime.tolist(),
        'value' : tmp_data.SEPSIS.tolist()
    }
    tmp_dict['SHOCK'] = {
        'starttime' : tmp_data.starttime.tolist(),
        'endtime' : tmp_data.endtime.tolist(),
        'value' : tmp_data.SHOCK.tolist()
    }

    return tmp_dict


# ==========================================================================================================================================================================================================================================
# ==========================================================================================================================================================================================================================================
# ==========================================================================================================================================================================================================================================

# tabularize ==============================================================================================================================================================================================================================
def tabularize(total_dict, step_length, item_dict, hadmid):
    data = total_dict[hadmid]
    item_dict = item_dict
    race = data['race']['value']
    gender = data['gender']['value']
    age = data['age']['value']
    
    # building timeline
    tmp_st = data['SHOCK']['starttime'][0]
    timeline = [tmp_st]
    
    timeline = [tmp_st]
    age = [age + (timeline[0] - data['age']['charttime'])/datetime.timedelta(days = 365)]
    while timeline[len(timeline)-1] < data['SHOCK']['endtime'][-1]:
        timeline.append(timeline[len(timeline)-1] + step_length)
        age.append(age[len(age)-1] + step_length/datetime.timedelta(days = 365))
    
    tmp_df = pd.DataFrame({
        'stay_id' : np.repeat(hadmid, len(timeline)-1),
        'seq_num' : list(range(len(timeline)-1)),
        'seq_ST' : timeline[0:len(timeline)-1],
        'seq_ET' : timeline[1:len(timeline)],
        'age' : age[0:len(timeline)-1],
        'gender': np.repeat(gender, len(timeline)-1),
        'race': np.repeat(race, len(timeline)-1)
    })
    seq_ST = tmp_df.seq_ST
    seq_ET = tmp_df.seq_ET

    # demo and vital
    tmp_item_data = pd.DataFrame({})
    tmp_item_list = item_dict['demo']+item_dict['vital']
    for i in range(len(tmp_item_list)):
        tmp_item = tmp_item_list[i]
        tmp_item_ct = pd.to_datetime(data[tmp_item]['charttime'])
        tmp_item_value = data[tmp_item]['value']
        tmp = pd.DataFrame({
            'tmp_item_ct': tmp_item_ct,
            'tmp_item_value': tmp_item_value
        })
        
        input = []
        input_value = []
        input_median = []
        input_max = []
        input_min = []
        presence = []
        for j in range(len(seq_ST)):
            if j == (len(seq_ST)-1):
                tmp_cond = (tmp.tmp_item_ct >= seq_ST[j]) & (tmp.tmp_item_ct <= seq_ET[j])
            else:
                tmp_cond = (tmp.tmp_item_ct >= seq_ST[j]) & (tmp.tmp_item_ct < seq_ET[j])

            if len(tmp.loc[tmp_cond].tmp_item_value) > 0:
                input.append(tmp.loc[tmp_cond].tmp_item_value.tolist())
                input_value.append(np.mean(tmp.loc[tmp_cond].tmp_item_value))
                input_median.append(np.median(tmp.loc[tmp_cond].tmp_item_value))
                input_max.append(np.max(tmp.loc[tmp_cond].tmp_item_value))
                input_min.append(np.min(tmp.loc[tmp_cond].tmp_item_value))
                presence.append(1)
            else:
                input.append(np.nan)
                input_value.append(np.nan)
                input_median.append(np.nan)
                input_max.append(np.nan)
                input_min.append(np.nan)
                presence.append(0)
        
        tmp_item_data[tmp_item_list[i]] = input
        tmp_item_data[tmp_item_list[i]+'_value'] = input_value
        tmp_item_data[tmp_item_list[i]+'_median'] = input_median
        tmp_item_data[tmp_item_list[i]+'_max'] = input_max
        tmp_item_data[tmp_item_list[i]+'_min'] = input_min
        tmp_item_data[tmp_item_list[i]+'_presence'] = presence
    tmp_item_data = tmp_item_data.interpolate(method='linear')
    tmp_item_data = tmp_item_data.fillna(method='bfill')

    # lab
    tmp_item_list = item_dict['lab']
    for i in range(len(tmp_item_list)):
        tmp_item = tmp_item_list[i]
        tmp_item_ct = pd.to_datetime(data[tmp_item]['charttime'])
        tmp_item_value = data[tmp_item]['value']
        tmp = pd.DataFrame({
            'tmp_item_ct': tmp_item_ct,
            'tmp_item_value': tmp_item_value
        })

        input = []
        input_value = []
        input_median = []
        input_max = []
        input_min = []
        presence = []
        for j in range(len(seq_ST)):
            if j == (len(seq_ST)-1):
                tmp_cond = (tmp.tmp_item_ct >= seq_ST[j]) & (tmp.tmp_item_ct <= seq_ET[j])
            else:
                tmp_cond = (tmp.tmp_item_ct >= seq_ST[j]) & (tmp.tmp_item_ct < seq_ET[j])

            if len(tmp.loc[tmp_cond].tmp_item_value) > 0:
                input.append(tmp.loc[tmp_cond].tmp_item_value.tolist())
                input_value.append(np.mean(tmp.loc[tmp_cond].tmp_item_value))
                input_median.append(np.median(tmp.loc[tmp_cond].tmp_item_value))
                input_max.append(np.max(tmp.loc[tmp_cond].tmp_item_value))
                input_min.append(np.min(tmp.loc[tmp_cond].tmp_item_value))
                presence.append(1)
            else:
                input.append(np.nan)
                input_value.append(np.nan)
                input_median.append(np.nan)
                input_max.append(np.nan)
                input_min.append(np.nan)
                presence.append(0)
        
        tmp_item_data[tmp_item_list[i]] = input
        tmp_item_data[tmp_item_list[i]+'_value'] = input_value
        tmp_item_data[tmp_item_list[i]+'_median'] = input_median
        tmp_item_data[tmp_item_list[i]+'_max'] = input_max
        tmp_item_data[tmp_item_list[i]+'_min'] = input_min
        tmp_item_data[tmp_item_list[i]+'_presence'] = presence
    tmp_item_data = tmp_item_data.fillna(method='ffill')
    tmp_item_data = tmp_item_data.fillna(method='bfill')

    # vasopressor--------------------------------------------------------------
    tmp_item_list = item_dict['vaso']
    for i in range(len(tmp_item_list)):
        tmp_item = tmp_item_list[i]
        tmp_item_value = data[tmp_item]['value']
        input_value = np.repeat(0.0, len(seq_ST)).tolist()
        input_rate = np.repeat(0.0, len(seq_ST)).tolist() # overestimate severity

        if tmp_item_value == []:# no value then move on to next loop
            tmp_item_data[tmp_item+'_value'] = input_value
            tmp_item_data[tmp_item+'_rate'] = input_rate
            continue

        tmp_item_rate = data[tmp_item]['rate']
        tmp_item_st = pd.to_datetime(data[tmp_item]['starttime'])
        tmp_item_et = pd.to_datetime(data[tmp_item]['endtime'])
        tmp = pd.DataFrame({
            'tmp_item_st': tmp_item_st,
            'tmp_item_et': tmp_item_et,
            'tmp_item_value': tmp_item_value,
            'tmp_item_rate': tmp_item_rate
        })
        tmp.tmp_item_value = tmp.tmp_item_value.astype(float)
        tmp.tmp_item_rate = tmp.tmp_item_rate.astype(float)

        for j in range(len(tmp_item_st)):
            for k in range(len(seq_ST)):
                if (tmp.tmp_item_st[j] >= seq_ST[k]) & (tmp.tmp_item_st[j] < seq_ET[k]):
                    tmp_ST_idx = k
                if (tmp.tmp_item_et[j] > seq_ST[k]) & (tmp.tmp_item_et[j] <= seq_ET[k]):
                    tmp_ET_idx = k
            if tmp_ST_idx == tmp_ET_idx:
                tmp_timeline = [tmp_item_st[j], tmp_item_et[j]]
            elif tmp_ST_idx+1 == tmp_ET_idx:
                tmp_timeline = [tmp_item_st[j]] + [seq_ST[tmp_ET_idx]] + [tmp_item_et[j]]
            else:
                tmp_timeline = [tmp_item_st[j]] + seq_ST[(tmp_ST_idx+1):(tmp_ET_idx+1)].tolist() + [tmp_item_et[j]]
            
            input_idx = tmp_ST_idx
            for k in range(len(tmp_timeline)-1):
                if tmp_item_st[j] == tmp_item_et[j]:
                    input_value[input_idx] = input_value[input_idx] + (tmp.tmp_item_value.tolist()[j])
                    if input_rate[input_idx] < tmp.tmp_item_rate[j]:
                        input_rate[input_idx] = tmp.tmp_item_rate[j]
                else:
                    tmp_ratio = (tmp_timeline[k+1] - tmp_timeline[k])/(tmp_item_et[j]-tmp_item_st[j])
                    input_value[input_idx] = input_value[input_idx]+ tmp_ratio*(tmp.tmp_item_value.tolist()[j])
                    if input_rate[input_idx] < tmp.tmp_item_rate[j]:
                        input_rate[input_idx] = tmp.tmp_item_rate[j]
                input_idx = input_idx+1

        tmp_item_data[tmp_item+'_value'] = input_value
        tmp_item_data[tmp_item+'_rate'] = input_rate

    
    # fluid
    tmp_item_list = item_dict['fluid']
    tmp_item = tmp_item_list[0]
    tmp_item_value = data[tmp_item]['value']
    tmp_item_st = pd.to_datetime(data[tmp_item]['starttime'])
    tmp_item_et = pd.to_datetime(data[tmp_item]['endtime'])
    tmp = pd.DataFrame({
        'tmp_item_st': tmp_item_st,
        'tmp_item_et': tmp_item_et,
        'tmp_item_value': tmp_item_value
    })
    tmp.tmp_item_value = tmp.tmp_item_value.astype(float)
    input_value = np.repeat(0.0, len(seq_ST)).tolist()

    if tmp_item_value == []:# no value then just put 0 values
        tmp_item_data[tmp_item+'_value'] = input_value
    else:
        for j in range(len(tmp_item_st)):
            for k in range(len(seq_ST)):
                if (tmp.tmp_item_st[j] >= seq_ST[k]) & (tmp.tmp_item_st[j] < seq_ET[k]):
                    tmp_ST_idx = k
                if (tmp.tmp_item_et[j] > seq_ST[k]) & (tmp.tmp_item_et[j] <= seq_ET[k]):
                    tmp_ET_idx = k
            if tmp_ST_idx == tmp_ET_idx:
                tmp_timeline = [tmp_item_st[j], tmp_item_et[j]]
            elif tmp_ST_idx+1 == tmp_ET_idx:
                tmp_timeline = [tmp_item_st[j]] + [seq_ST[tmp_ET_idx]] + [tmp_item_et[j]]
            else:
                tmp_timeline = [tmp_item_st[j]] + seq_ST[(tmp_ST_idx+1):(tmp_ET_idx+1)].tolist()+ [tmp_item_et[j]]
            
            input_idx = tmp_ST_idx
            for k in range(len(tmp_timeline)-1):
                if tmp_item_st[j] == tmp_item_et[j]:
                    input_value[input_idx] = input_value[input_idx] + (tmp.tmp_item_value.tolist()[j])
                else:
                    tmp_ratio = (tmp_timeline[k+1] - tmp_timeline[k])/(tmp_item_et[j]-tmp_item_st[j])
                    input_value[input_idx] = input_value[input_idx] + tmp_ratio*(tmp.tmp_item_value.tolist()[j])
                input_idx = input_idx + 1

        tmp_item_data[tmp_item+'_value'] = input_value

    # urine
    tmp_item_list = item_dict['urine']
    for i in range(len(tmp_item_list)):
        tmp_item = tmp_item_list[i]
        tmp_item_ct = pd.to_datetime(data[tmp_item]['charttime'])
        tmp_item_value = data[tmp_item]['value']
        tmp = pd.DataFrame({
            'tmp_item_ct': tmp_item_ct,
            'tmp_item_value': tmp_item_value
        })

        input = []
        input_value = []
        for j in range(len(seq_ST)):
            if j == (len(seq_ST)-1):
                tmp_cond = (tmp.tmp_item_ct >= seq_ST[j]) & (tmp.tmp_item_ct <= seq_ET[j])
            else:
                tmp_cond = (tmp.tmp_item_ct >= seq_ST[j]) & (tmp.tmp_item_ct < seq_ET[j])

            if len(tmp.loc[tmp_cond].tmp_item_value) > 0:
                input.append(tmp.loc[tmp_cond].tmp_item_value.tolist())
                input_value.append(np.nansum(tmp.loc[tmp_cond].tmp_item_value))
            else:
                input.append(np.nan)
                input_value.append(0)
        
        tmp_item_data[tmp_item_list[i]] = input
        tmp_item_data[tmp_item_list[i]+'_value'] = input_value

    # vent
    tmp_item_list = item_dict['vent']
    tmp_item = tmp_item_list[0]
    tmp_item_value = data[tmp_item]['value']
    tmp_item_st = pd.to_datetime(data[tmp_item]['starttime'])
    tmp_item_et = pd.to_datetime(data[tmp_item]['endtime'])
    tmp = pd.DataFrame({
        'tmp_item_st': tmp_item_st,
        'tmp_item_et': tmp_item_et
    })
    input_value = np.repeat(0, len(seq_ST)).tolist()

    if tmp_item_value == []:# no value then just put 0 values
        tmp_item_data[tmp_item+'_value'] = input_value
    else:
        for j in range(len(tmp_item_st)):
            for k in range(len(seq_ST)):
                if (tmp.tmp_item_st[j] >= seq_ST[k]) & (tmp.tmp_item_st[j] < seq_ET[k]):
                    tmp_ST_idx = k
                if (tmp.tmp_item_et[j] > seq_ST[k]) & (tmp.tmp_item_et[j] <= seq_ET[k]):
                    tmp_ET_idx = k
            if tmp_ST_idx == tmp_ET_idx:
                tmp_timeline = [tmp_item_st[j], tmp_item_et[j]]
            elif tmp_ST_idx+1 == tmp_ET_idx:
                tmp_timeline = [tmp_item_st[j]] + [seq_ST[tmp_ET_idx]] + [tmp_item_et[j]]
            else:
                tmp_timeline = [tmp_item_st[j]] + seq_ST[(tmp_ST_idx+1):(tmp_ET_idx+1)].tolist() + [tmp_item_et[j]]
            
            input_idx = tmp_ST_idx
            for k in range(len(tmp_timeline)-1):
                input_value[input_idx] = 1
                input_idx = input_idx + 1

        tmp_item_data[tmp_item+'_value'] = input_value
    
    tmp_item_data = tmp_item_data.fillna(0)
    tmp_df = pd.concat([tmp_df, tmp_item_data], axis = 1)

    # bmi calculation
    tmp_df['bmi'] = (tmp_df.weight_value)/(tmp_df.height_value*tmp_df.height_value)*10000
    

    # labeling-----------------------------------------------------------------------------------------------------------------------------------------
    # CNS_SOFA
    tmp_sofa = np.repeat(0, tmp_df.shape[0])
    tmp_GCS = tmp_df.gcs_value.tolist()

    for i in range(len(tmp_GCS)):
        if tmp_GCS[i] < 6:
            tmp_sofa[i] = 4
        elif (tmp_GCS[i] >= 6) & (tmp_GCS[i] <= 9):
            tmp_sofa[i] = 3
        elif (tmp_GCS[i] >= 10) & (tmp_GCS[i] <= 12):
            tmp_sofa[i] = 2
        elif (tmp_GCS[i] >= 13) & (tmp_GCS[i] <= 14):
            tmp_sofa[i] = 1
    
    tmp_df['CNS_SOFA'] = tmp_sofa
    

    # CARDIO_SOFA
    tmp_sofa = np.repeat(0, tmp_df.shape[0])
    tmp_MAP = tmp_df['map_value'].tolist()
    tmp_epi_rate = tmp_df['epinephrine_rate'].tolist() 
    tmp_dop_rate = tmp_df['dopamine_rate'].tolist()
    tmp_dob_rate = tmp_df['dobutamine_rate'].tolist()
    tmp_nor_rate = tmp_df['norepinephrine_rate'].tolist()

    for i in range(len(tmp_sofa)):
        if tmp_MAP[i] < 70:
            tmp_sofa[i] = 1
            if (tmp_dop_rate[i] > 0) | (tmp_dob_rate[i] > 0):
                tmp_sofa[i] = 2
            if (tmp_dop_rate[i] > 5) | ((tmp_epi_rate[i] > 0) & (tmp_epi_rate[i] <= 0.1))| ((tmp_nor_rate[i] > 0) & (tmp_nor_rate[i] <= 0.1)):
                tmp_sofa[i] = 3
            if (tmp_dop_rate[i] > 15) | (tmp_epi_rate[i] > 0.1) | (tmp_nor_rate[i] > 0.1):
                tmp_sofa[i] = 4
            
    tmp_df['CARDIO_SOFA'] = tmp_sofa
    tmp_cond = (tmp_df['epinephrine_rate']>0) | (tmp_df['dopamine_rate']>0) | (tmp_df['dobutamine_rate']>0) | (tmp_df['norepinephrine_rate']>0) | (tmp_df['phenylephrine_rate']>0) | (tmp_df['vasopressin_rate']>0)
    tmp_df['vaso_presence'] = np.where(tmp_cond, 1, 0)


    # RESP_SOFA
    tmp_sofa = np.repeat(0, tmp_df.shape[0])
    tmp_paO2 = tmp_df['pao2_value'].tolist()
    tmp_fiO2 = tmp_df['fio2_value'].tolist()
    tmp_vent = tmp_df['ventilator_value'].tolist()

    for i in range(len(tmp_sofa)):
        if tmp_vent[i] == 0:
            if ((tmp_paO2[i]/tmp_fiO2[i]) < 400) & ((tmp_paO2[i]/tmp_fiO2[i]) >= 300):
                tmp_sofa[i] = 1
            elif ((tmp_paO2[i]/tmp_fiO2[i]) < 300) & ((tmp_paO2[i]/tmp_fiO2[i]) >= 200):
                tmp_sofa[i] = 2
        else:
            if ((tmp_paO2[i]/tmp_fiO2[i]) < 200) & ((tmp_paO2[i]/tmp_fiO2[i]) >= 100):
                tmp_sofa[i] = 3
            elif ((tmp_paO2[i]/tmp_fiO2[i]) < 100) & ((tmp_paO2[i]/tmp_fiO2[i]) >= 0):
                tmp_sofa[i] = 4

    tmp_df['RESP_SOFA'] = tmp_sofa


    # COAG_SOFA
    tmp_sofa = np.repeat(0, tmp_df.shape[0])
    tmp_plat = tmp_df['platelets_value'].tolist()

    for i in range(len(tmp_plat)):
        if (tmp_plat[i] >= 100) & (tmp_plat[i] < 150):
            tmp_sofa[i] = 1
        elif (tmp_plat[i] >= 50) & (tmp_plat[i] < 100):
            tmp_sofa[i] = 2
        elif (tmp_plat[i] >= 20) & (tmp_plat[i] < 50):
            tmp_sofa[i] = 3
        elif (tmp_plat[i] < 20):
            tmp_sofa[i] = 4
    
    tmp_df['COAG_SOFA'] = tmp_sofa


    # LIVER_SOFA
    tmp_sofa = np.repeat(0, tmp_df.shape[0])
    tmp_bili = tmp_df.bilirubin_value.tolist()

    for i in range(len(tmp_bili)):
        if (tmp_bili[i] >= 1.2) & (tmp_bili[i] < 2):
            tmp_sofa[i] = 1
        elif (tmp_bili[i] >= 2) & (tmp_bili[i] < 6):
            tmp_sofa[i] = 2
        elif (tmp_bili[i] >= 6) & (tmp_bili[i] < 12):
            tmp_sofa[i] = 3
        elif (tmp_bili[i] >= 12):
            tmp_sofa[i] = 4
    
    tmp_df['LIVER_SOFA'] = tmp_sofa


    # RENAL_SOFA
    tmp_sofa = np.repeat(0, tmp_df.shape[0])
    tmp_creat = tmp_df.creatinine_value.tolist()

    for i in range(len(tmp_creat)):
        if (tmp_creat[i] >= 1.2) & (tmp_creat[i] < 2):
            tmp_sofa[i] = 1
        elif (tmp_creat[i] >= 2) & (tmp_creat[i] < 3.5):
            tmp_sofa[i] = 2
        elif (tmp_creat[i] >= 3.5) & (tmp_creat[i] < 5):
            tmp_sofa[i] = 3
        elif (tmp_creat[i] >= 5):
            tmp_sofa[i] = 4
    
    tmp_df['RENAL_SOFA'] = tmp_sofa

    # SOFA
    tmp_df['SOFA'] = tmp_df['CNS_SOFA'] + tmp_df['CARDIO_SOFA'] + tmp_df['RESP_SOFA'] + tmp_df['COAG_SOFA'] + tmp_df['LIVER_SOFA'] + tmp_df['RENAL_SOFA']

    # SEPSIS
    tmp_df['SEPSIS'] = np.where(tmp_df['SOFA']>=2, 1, 0)

    # SEPTIC SHOCK
    tmp_df['SHOCK'] = np.where((tmp_df['lactate_value']>=2)&(tmp_df['vaso_presence']==1), 1, 0)

    tmp_data = tmp_df#.to_dict()
    return tmp_data
    
    