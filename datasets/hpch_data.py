import random
import torch
import pandas as pd
from pathlib import Path
import numpy as np
import torch.utils.data as data
from torch.utils.data import dataloader
from loguru import logger
from sklearn.model_selection import StratifiedKFold


class HpchData(data.Dataset):
    def __init__(self, dataset_cfg=None, state=None):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg
        self.need_rad_feats =  [
        'FrequencySize', 'MaxIntensity', 'MeanDeviation', 'MeanValue',
       'MedianIntensity', 'MinIntensity', 'Percentile10', 'Percentile15',
       'Percentile20', 'Percentile25', 'Percentile30', 'Percentile35',
       'Percentile40', 'Percentile45', 'Percentile5', 'Percentile50',
       'Percentile55', 'Percentile60', 'Percentile65', 'Percentile70',
       'Percentile75', 'Percentile80', 'Percentile85', 'Percentile90',
       'Percentile95', 'Quantile0.025', 'Quantile0.25', 'Quantile0.5',
       'Quantile0.75', 'Quantile0.975', 'RMS', 'Range',
       'Relat4eDeviation', 'Variance', 'VolumeCount', 'VoxelValueSum',
       'histogramEnergy', 'histogramEntropy', 'kurtosis', 'skewness',
       'stdDeviation', 'uniformity',
       'ClusterProminence_AllDirection_offset1',
       'ClusterProminence_AllDirection_offset1_SD',
       'ClusterProminence_AllDirection_offset4',
       'ClusterProminence_AllDirection_offset4_SD',
       'ClusterProminence_AllDirection_offset7',
       'ClusterProminence_AllDirection_offset7_SD',
       'ClusterProminence_angle0_offset1',
       'ClusterProminence_angle0_offset4',
       'ClusterProminence_angle0_offset7',
       'ClusterProminence_angle135_offset1',
       'ClusterProminence_angle135_offset4',
       'ClusterProminence_angle135_offset7',
       'ClusterProminence_angle45_offset1',
       'ClusterProminence_angle45_offset4',
       'ClusterProminence_angle45_offset7',
       'ClusterProminence_angle90_offset1',
       'ClusterProminence_angle90_offset4',
       'ClusterProminence_angle90_offset7',
       'ClusterShade_AllDirection_offset1',
       'ClusterShade_AllDirection_offset1_SD',
       'ClusterShade_AllDirection_offset4',
       'ClusterShade_AllDirection_offset4_SD',
       'ClusterShade_AllDirection_offset7',
       'ClusterShade_AllDirection_offset7_SD',
       'ClusterShade_angle0_offset1', 'ClusterShade_angle0_offset4',
       'ClusterShade_angle0_offset7', 'ClusterShade_angle135_offset1',
       'ClusterShade_angle135_offset4', 'ClusterShade_angle135_offset7',
       'ClusterShade_angle45_offset1', 'ClusterShade_angle45_offset4',
       'ClusterShade_angle45_offset7', 'ClusterShade_angle90_offset1',
       'ClusterShade_angle90_offset4', 'ClusterShade_angle90_offset7',
       'Correlation_AllDirection_offset1',
       'Correlation_AllDirection_offset1_SD',
       'Correlation_AllDirection_offset4',
       'Correlation_AllDirection_offset4_SD',
       'Correlation_AllDirection_offset7',
       'Correlation_AllDirection_offset7_SD',
       'Correlation_angle0_offset1', 'Correlation_angle0_offset4',
       'Correlation_angle0_offset7', 'Correlation_angle135_offset1',
       'Correlation_angle135_offset4', 'Correlation_angle135_offset7',
       'Correlation_angle45_offset1', 'Correlation_angle45_offset4',
       'Correlation_angle45_offset7', 'Correlation_angle90_offset1',
       'Correlation_angle90_offset4', 'Correlation_angle90_offset7',
       'GLCMEnergy_AllDirection_offset1',
       'GLCMEnergy_AllDirection_offset1_SD',
       'GLCMEnergy_AllDirection_offset4',
       'GLCMEnergy_AllDirection_offset4_SD',
       'GLCMEnergy_AllDirection_offset7',
       'GLCMEnergy_AllDirection_offset7_SD', 'GLCMEnergy_angle0_offset1',
       'GLCMEnergy_angle0_offset4', 'GLCMEnergy_angle0_offset7',
       'GLCMEnergy_angle135_offset1', 'GLCMEnergy_angle135_offset4',
       'GLCMEnergy_angle135_offset7', 'GLCMEnergy_angle45_offset1',
       'GLCMEnergy_angle45_offset4', 'GLCMEnergy_angle45_offset7',
       'GLCMEnergy_angle90_offset1', 'GLCMEnergy_angle90_offset4',
       'GLCMEnergy_angle90_offset7', 'GLCMEntropy_AllDirection_offset1',
       'GLCMEntropy_AllDirection_offset1_SD',
       'GLCMEntropy_AllDirection_offset4',
       'GLCMEntropy_AllDirection_offset4_SD',
       'GLCMEntropy_AllDirection_offset7',
       'GLCMEntropy_AllDirection_offset7_SD',
       'GLCMEntropy_angle0_offset1', 'GLCMEntropy_angle0_offset4',
       'GLCMEntropy_angle0_offset7', 'GLCMEntropy_angle135_offset1',
       'GLCMEntropy_angle135_offset4', 'GLCMEntropy_angle135_offset7',
       'GLCMEntropy_angle45_offset1', 'GLCMEntropy_angle45_offset4',
       'GLCMEntropy_angle45_offset7', 'GLCMEntropy_angle90_offset1',
       'GLCMEntropy_angle90_offset4', 'GLCMEntropy_angle90_offset7',
       'HaralickCorrelation_AllDirection_offset1',
       'HaralickCorrelation_AllDirection_offset1_SD',
       'HaralickCorrelation_AllDirection_offset4',
       'HaralickCorrelation_AllDirection_offset4_SD',
       'HaralickCorrelation_AllDirection_offset7',
       'HaralickCorrelation_AllDirection_offset7_SD',
       'HaralickCorrelation_angle0_offset1',
       'HaralickCorrelation_angle0_offset4',
       'HaralickCorrelation_angle0_offset7',
       'HaralickCorrelation_angle135_offset1',
       'HaralickCorrelation_angle135_offset4',
       'HaralickCorrelation_angle135_offset7',
       'HaralickCorrelation_angle45_offset1',
       'HaralickCorrelation_angle45_offset4',
       'HaralickCorrelation_angle45_offset7',
       'HaralickCorrelation_angle90_offset1',
       'HaralickCorrelation_angle90_offset4',
       'HaralickCorrelation_angle90_offset7',
       'Inertia_AllDirection_offset1', 'Inertia_AllDirection_offset1_SD',
       'Inertia_AllDirection_offset4', 'Inertia_AllDirection_offset4_SD',
       'Inertia_AllDirection_offset7', 'Inertia_AllDirection_offset7_SD',
       'Inertia_angle0_offset1', 'Inertia_angle0_offset4',
       'Inertia_angle0_offset7', 'Inertia_angle135_offset1',
       'Inertia_angle135_offset4', 'Inertia_angle135_offset7',
       'Inertia_angle45_offset1', 'Inertia_angle45_offset4',
       'Inertia_angle45_offset7', 'Inertia_angle90_offset1',
       'Inertia_angle90_offset4', 'Inertia_angle90_offset7',
       'InverseDifferenceMoment_AllDirection_offset1',
       'InverseDifferenceMoment_AllDirection_offset1_SD',
       'InverseDifferenceMoment_AllDirection_offset4',
       'InverseDifferenceMoment_AllDirection_offset4_SD',
       'InverseDifferenceMoment_AllDirection_offset7',
       'InverseDifferenceMoment_AllDirection_offset7_SD',
       'InverseDifferenceMoment_angle0_offset1',
       'InverseDifferenceMoment_angle0_offset4',
       'InverseDifferenceMoment_angle0_offset7',
       'InverseDifferenceMoment_angle135_offset1',
       'InverseDifferenceMoment_angle135_offset4',
       'InverseDifferenceMoment_angle135_offset7',
       'InverseDifferenceMoment_angle45_offset1',
       'InverseDifferenceMoment_angle45_offset4',
       'InverseDifferenceMoment_angle45_offset7',
       'InverseDifferenceMoment_angle90_offset1',
       'InverseDifferenceMoment_angle90_offset4',
       'InverseDifferenceMoment_angle90_offset7', 'AngularSecondMoment',
       'HaraEntroy', 'HaraVariance', 'contrast', 'differenceEntropy',
       'differenceVariance', 'inverseDifferenceMoment', 'sumAverage',
       'sumEntropy', 'sumVariance',
       'GreyLevelNonuniformity_AllDirection_offset1',
       'GreyLevelNonuniformity_AllDirection_offset1_SD',
       'GreyLevelNonuniformity_AllDirection_offset4',
       'GreyLevelNonuniformity_AllDirection_offset4_SD',
       'GreyLevelNonuniformity_AllDirection_offset7',
       'GreyLevelNonuniformity_AllDirection_offset7_SD',
       'GreyLevelNonuniformity_angle0_offset1',
       'GreyLevelNonuniformity_angle0_offset4',
       'GreyLevelNonuniformity_angle0_offset7',
       'GreyLevelNonuniformity_angle135_offset1',
       'GreyLevelNonuniformity_angle135_offset4',
       'GreyLevelNonuniformity_angle135_offset7',
       'GreyLevelNonuniformity_angle45_offset1',
       'GreyLevelNonuniformity_angle45_offset4',
       'GreyLevelNonuniformity_angle45_offset7',
       'GreyLevelNonuniformity_angle90_offset1',
       'GreyLevelNonuniformity_angle90_offset4',
       'GreyLevelNonuniformity_angle90_offset7',
       'HighGreyLevelRunEmphasis_AllDirection_offset1',
       'HighGreyLevelRunEmphasis_AllDirection_offset1_SD',
       'HighGreyLevelRunEmphasis_AllDirection_offset4',
       'HighGreyLevelRunEmphasis_AllDirection_offset4_SD',
       'HighGreyLevelRunEmphasis_AllDirection_offset7',
       'HighGreyLevelRunEmphasis_AllDirection_offset7_SD',
       'HighGreyLevelRunEmphasis_angle0_offset1',
       'HighGreyLevelRunEmphasis_angle0_offset4',
       'HighGreyLevelRunEmphasis_angle0_offset7',
       'HighGreyLevelRunEmphasis_angle135_offset1',
       'HighGreyLevelRunEmphasis_angle135_offset4',
       'HighGreyLevelRunEmphasis_angle135_offset7',
       'HighGreyLevelRunEmphasis_angle45_offset1',
       'HighGreyLevelRunEmphasis_angle45_offset4',
       'HighGreyLevelRunEmphasis_angle45_offset7',
       'HighGreyLevelRunEmphasis_angle90_offset1',
       'HighGreyLevelRunEmphasis_angle90_offset4',
       'HighGreyLevelRunEmphasis_angle90_offset7',
       'LongRunEmphasis_AllDirection_offset1',
       'LongRunEmphasis_AllDirection_offset1_SD',
       'LongRunEmphasis_AllDirection_offset4',
       'LongRunEmphasis_AllDirection_offset4_SD',
       'LongRunEmphasis_AllDirection_offset7',
       'LongRunEmphasis_AllDirection_offset7_SD',
       'LongRunEmphasis_angle0_offset1', 'LongRunEmphasis_angle0_offset4',
       'LongRunEmphasis_angle0_offset7',
       'LongRunEmphasis_angle135_offset1',
       'LongRunEmphasis_angle135_offset4',
       'LongRunEmphasis_angle135_offset7',
       'LongRunEmphasis_angle45_offset1',
       'LongRunEmphasis_angle45_offset4',
       'LongRunEmphasis_angle45_offset7',
       'LongRunEmphasis_angle90_offset1',
       'LongRunEmphasis_angle90_offset4',
       'LongRunEmphasis_angle90_offset7',
       'LongRunHighGreyLevelEmphasis_AllDirection_offset1',
       'LongRunHighGreyLevelEmphasis_AllDirection_offset1_SD',
       'LongRunHighGreyLevelEmphasis_AllDirection_offset4',
       'LongRunHighGreyLevelEmphasis_AllDirection_offset4_SD',
       'LongRunHighGreyLevelEmphasis_AllDirection_offset7',
       'LongRunHighGreyLevelEmphasis_AllDirection_offset7_SD',
       'LongRunHighGreyLevelEmphasis_angle0_offset1',
       'LongRunHighGreyLevelEmphasis_angle0_offset4',
       'LongRunHighGreyLevelEmphasis_angle0_offset7',
       'LongRunHighGreyLevelEmphasis_angle135_offset1',
       'LongRunHighGreyLevelEmphasis_angle135_offset4',
       'LongRunHighGreyLevelEmphasis_angle135_offset7',
       'LongRunHighGreyLevelEmphasis_angle45_offset1',
       'LongRunHighGreyLevelEmphasis_angle45_offset4',
       'LongRunHighGreyLevelEmphasis_angle45_offset7',
       'LongRunHighGreyLevelEmphasis_angle90_offset1',
       'LongRunHighGreyLevelEmphasis_angle90_offset4',
       'LongRunHighGreyLevelEmphasis_angle90_offset7',
       'LongRunLowGreyLevelEmphasis_AllDirection_offset1',
       'LongRunLowGreyLevelEmphasis_AllDirection_offset1_SD',
       'LongRunLowGreyLevelEmphasis_AllDirection_offset4',
       'LongRunLowGreyLevelEmphasis_AllDirection_offset4_SD',
       'LongRunLowGreyLevelEmphasis_AllDirection_offset7',
       'LongRunLowGreyLevelEmphasis_AllDirection_offset7_SD',
       'LongRunLowGreyLevelEmphasis_angle0_offset1',
       'LongRunLowGreyLevelEmphasis_angle0_offset4',
       'LongRunLowGreyLevelEmphasis_angle0_offset7',
       'LongRunLowGreyLevelEmphasis_angle135_offset1',
       'LongRunLowGreyLevelEmphasis_angle135_offset4',
       'LongRunLowGreyLevelEmphasis_angle135_offset7',
       'LongRunLowGreyLevelEmphasis_angle45_offset1',
       'LongRunLowGreyLevelEmphasis_angle45_offset4',
       'LongRunLowGreyLevelEmphasis_angle45_offset7',
       'LongRunLowGreyLevelEmphasis_angle90_offset1',
       'LongRunLowGreyLevelEmphasis_angle90_offset4',
       'LongRunLowGreyLevelEmphasis_angle90_offset7',
       'LowGreyLevelRunEmphasis_AllDirection_offset1',
       'LowGreyLevelRunEmphasis_AllDirection_offset1_SD',
       'LowGreyLevelRunEmphasis_AllDirection_offset4',
       'LowGreyLevelRunEmphasis_AllDirection_offset4_SD',
       'LowGreyLevelRunEmphasis_AllDirection_offset7',
       'LowGreyLevelRunEmphasis_AllDirection_offset7_SD',
       'LowGreyLevelRunEmphasis_angle0_offset1',
       'LowGreyLevelRunEmphasis_angle0_offset4',
       'LowGreyLevelRunEmphasis_angle0_offset7',
       'LowGreyLevelRunEmphasis_angle135_offset1',
       'LowGreyLevelRunEmphasis_angle135_offset4',
       'LowGreyLevelRunEmphasis_angle135_offset7',
       'LowGreyLevelRunEmphasis_angle45_offset1',
       'LowGreyLevelRunEmphasis_angle45_offset4',
       'LowGreyLevelRunEmphasis_angle45_offset7',
       'LowGreyLevelRunEmphasis_angle90_offset1',
       'LowGreyLevelRunEmphasis_angle90_offset4',
       'LowGreyLevelRunEmphasis_angle90_offset7',
       'RunLengthNonuniformity_AllDirection_offset1',
       'RunLengthNonuniformity_AllDirection_offset1_SD',
       'RunLengthNonuniformity_AllDirection_offset4',
       'RunLengthNonuniformity_AllDirection_offset4_SD',
       'RunLengthNonuniformity_AllDirection_offset7',
       'RunLengthNonuniformity_AllDirection_offset7_SD',
       'RunLengthNonuniformity_angle0_offset1',
       'RunLengthNonuniformity_angle0_offset4',
       'RunLengthNonuniformity_angle0_offset7',
       'RunLengthNonuniformity_angle135_offset1',
       'RunLengthNonuniformity_angle135_offset4',
       'RunLengthNonuniformity_angle135_offset7',
       'RunLengthNonuniformity_angle45_offset1',
       'RunLengthNonuniformity_angle45_offset4',
       'RunLengthNonuniformity_angle45_offset7',
       'RunLengthNonuniformity_angle90_offset1',
       'RunLengthNonuniformity_angle90_offset4',
       'RunLengthNonuniformity_angle90_offset7',
       'ShortRunEmphasis_AllDirection_offset1',
       'ShortRunEmphasis_AllDirection_offset1_SD',
       'ShortRunEmphasis_AllDirection_offset4',
       'ShortRunEmphasis_AllDirection_offset4_SD',
       'ShortRunEmphasis_AllDirection_offset7',
       'ShortRunEmphasis_AllDirection_offset7_SD',
       'ShortRunEmphasis_angle0_offset1',
       'ShortRunEmphasis_angle0_offset4',
       'ShortRunEmphasis_angle0_offset7',
       'ShortRunEmphasis_angle135_offset1',
       'ShortRunEmphasis_angle135_offset4',
       'ShortRunEmphasis_angle135_offset7',
       'ShortRunEmphasis_angle45_offset1',
       'ShortRunEmphasis_angle45_offset4',
       'ShortRunEmphasis_angle45_offset7',
       'ShortRunEmphasis_angle90_offset1',
       'ShortRunEmphasis_angle90_offset4',
       'ShortRunEmphasis_angle90_offset7',
       'ShortRunHighGreyLevelEmphasis_AllDirection_offset1',
       'ShortRunHighGreyLevelEmphasis_AllDirection_offset1_SD',
       'ShortRunHighGreyLevelEmphasis_AllDirection_offset4',
       'ShortRunHighGreyLevelEmphasis_AllDirection_offset4_SD',
       'ShortRunHighGreyLevelEmphasis_AllDirection_offset7',
       'ShortRunHighGreyLevelEmphasis_AllDirection_offset7_SD',
       'ShortRunHighGreyLevelEmphasis_angle0_offset1',
       'ShortRunHighGreyLevelEmphasis_angle0_offset4',
       'ShortRunHighGreyLevelEmphasis_angle0_offset7',
       'ShortRunHighGreyLevelEmphasis_angle135_offset1',
       'ShortRunHighGreyLevelEmphasis_angle135_offset4',
       'ShortRunHighGreyLevelEmphasis_angle135_offset7',
       'ShortRunHighGreyLevelEmphasis_angle45_offset1',
       'ShortRunHighGreyLevelEmphasis_angle45_offset4',
       'ShortRunHighGreyLevelEmphasis_angle45_offset7',
       'ShortRunHighGreyLevelEmphasis_angle90_offset1',
       'ShortRunHighGreyLevelEmphasis_angle90_offset4',
       'ShortRunHighGreyLevelEmphasis_angle90_offset7',
       'ShortRunLowGreyLevelEmphasis_AllDirection_offset1',
       'ShortRunLowGreyLevelEmphasis_AllDirection_offset1_SD',
       'ShortRunLowGreyLevelEmphasis_AllDirection_offset4',
       'ShortRunLowGreyLevelEmphasis_AllDirection_offset4_SD',
       'ShortRunLowGreyLevelEmphasis_AllDirection_offset7',
       'ShortRunLowGreyLevelEmphasis_AllDirection_offset7_SD',
       'ShortRunLowGreyLevelEmphasis_angle0_offset1',
       'ShortRunLowGreyLevelEmphasis_angle0_offset4',
       'ShortRunLowGreyLevelEmphasis_angle0_offset7',
       'ShortRunLowGreyLevelEmphasis_angle135_offset1',
       'ShortRunLowGreyLevelEmphasis_angle135_offset4',
       'ShortRunLowGreyLevelEmphasis_angle135_offset7',
       'ShortRunLowGreyLevelEmphasis_angle45_offset1',
       'ShortRunLowGreyLevelEmphasis_angle45_offset4',
       'ShortRunLowGreyLevelEmphasis_angle45_offset7',
       'ShortRunLowGreyLevelEmphasis_angle90_offset1',
       'ShortRunLowGreyLevelEmphasis_angle90_offset4',
       'ShortRunLowGreyLevelEmphasis_angle90_offset7', 'Compactness1',
       'Compactness2', 'Maximum3DDiameter', 'SphericalDisproportion',
       'Sphericity', 'SurfaceArea', 'SurfaceVolumeRatio', 'VolumeCC',
       'VolumeMM', ' SizeZoneVariability', 'HighIntensityEmphasis',
       'HighIntensityLargeAreaEmphasis', 'HighIntensitySmallAreaEmphasis',
       'IntensityVariability', 'LargeAreaEmphasis',
       'LowIntensityEmphasis', 'LowIntensityLargeAreaEmphasis',
       'LowIntensitySmallAreaEmphasis', 'SmallAreaEmphasis',
       'ZonePercentage',]

        self.need_Crad_feats = ['C_' + f for f in self.need_rad_feats]

        # #---->data和label
        self.feature_dir = self.dataset_cfg.data_dir
        self.csv_dir = self.dataset_cfg.label_dir + f'all_data_rad_v2.csv'
        self.slide_data_ = pd.read_csv(self.csv_dir, index_col=0).reset_index()

        feature_names = [f for f in self.slide_data_.columns]

        if self.dataset_cfg.label_name == 'OS':
            print(f'{state} : use os as label')
            if state == 'train':
                col_dict = {'os_disc_label': 'disc_label', 'os_censorship': 'censorship', 'os_survival_months': 'survival_months'}
                self.slide_data_.rename(columns=col_dict, inplace=True)
            elif state == 'val':
                col_dict = {'os_disc_label': 'disc_label', 'os_censorship': 'censorship', 'os_survival_months': 'survival_months'}
                self.slide_data_.rename(columns=col_dict, inplace=True)
            elif state == 'test':
                col_dict = {'os_disc_label': 'disc_label', 'os_censorship': 'censorship', 'os_survival_months': 'survival_months'}
                self.slide_data_.rename(columns=col_dict, inplace=True)
            else:
                raise ValueError('Invalid state!')
        else:
            print(f'{state} : use pfs as label')
            if state == 'train':
                col_dict = {'pfs_disc_label': 'disc_label', 'pfs_censorship': 'censorship', 'pfs_survival_months': 'survival_months'}
                self.slide_data_.rename(columns=col_dict, inplace=True)
            elif state == 'val':
                col_dict = {'pfs_disc_label': 'disc_label', 'pfs_censorship': 'censorship', 'pfs_survival_months': 'survival_months'}
                self.slide_data_.rename(columns=col_dict, inplace=True)
            elif state == 'test':
                col_dict = {'pfs_disc_label': 'disc_label', 'pfs_censorship': 'censorship', 'pfs_survival_months': 'survival_months'}
                self.slide_data_.rename(columns=col_dict, inplace=True)
            else:
                raise ValueError('Invalid state!')
        if state == 'train':
            self.slide_data = self.slide_data_[self.slide_data_['Group'] == 'Train'].reset_index(drop=True)
        elif state == 'val':
            self.slide_data = self.slide_data_[self.slide_data_['Group'] != 'Train'].reset_index(drop=True)
        elif state == 'test':
            self.slide_data = self.slide_data_[self.slide_data_['Group'] != 'Train'].reset_index(drop=True)

        self.slide_id = self.slide_data['slide_id'].dropna()
        self.survival_months = self.slide_data['survival_months'].dropna()
        self.censorship = self.slide_data['censorship'].dropna()
        self.case_id = self.slide_data['case_id'].dropna()
        self.label = self.slide_data['disc_label'].dropna()

        # #---->Concat related information together
        splits = [self.slide_id, self.survival_months, self.censorship, self.case_id, self.label]
        self.split_data = pd.concat(splits, ignore_index = True, axis=1)
        self.split_data.columns = ['slide_id', 'survival_months', 'censorship', 'case_id', 'disc_label']

        #---->get patient data
        self.patient_df = self.split_data.drop_duplicates(['case_id']).copy()
        self.patient_df.set_index(keys='case_id', drop=True, inplace=True)
        self.split_data.set_index(keys='case_id', drop=True, inplace=True)

        #---->Establish a connection between patient_df and data
        self.patient_dict = {}
        for patient in self.patient_df.index:
            slide_ids = self.split_data.loc[patient, 'slide_id'] #取出case_id
            slide_ids = [slide_ids]
            self.patient_dict.update({patient:slide_ids}) #更新字典，每个patient包括哪些Slide
    def __len__(self):
        return len(self.patient_df)

    def __getitem__(self, idx):
        case_id = self.case_id[idx]
        
        event_time = self.survival_months[idx]
        censorship = self.censorship[idx]
        label = self.label[idx]
        slide_ids = self.patient_dict[case_id]

        features = []
        features_rad = []
        features_Crad = []
        for slide_id in slide_ids:
            full_path = Path(self.feature_dir) / f'{slide_id}.pt'
            features.append(torch.load(full_path))

            rad_feats = torch.from_numpy(self.slide_data_[self.need_rad_feats].loc[self.slide_data_['slide_id'] == slide_id].values.astype(float))
            features_rad.append(rad_feats)

            Crad_feats = torch.from_numpy(self.slide_data_[self.need_rad_feats].loc[self.slide_data_['slide_id'] == slide_id].values.astype(float))
            features_Crad.append(Crad_feats)

        features = torch.cat(features, dim=0)
        features_rad = torch.cat(features_rad, dim=0)
        features_Crad = torch.cat(features_Crad, dim=0)


        return case_id, features, features_rad, features_Crad, label, event_time, censorship