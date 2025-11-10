from .statistics_extractor import StatisticsExtractor
import json
import numpy as np
import pandas as pd

class MulticlassStatisticsExtractor(StatisticsExtractor):
    def __init__(self, model_name, data_dir, test_csv_dir, test_csv_filenames, gt_json_filename, img_class_txt, split_classes_file, img_to_exclude_txt=None):

        super().__init__(model_name, data_dir, test_csv_dir, test_csv_filenames, gt_json_filename, img_class_txt, split_classes_file, img_to_exclude_txt=img_to_exclude_txt)
        

    def load_data(self):
        
        self.df_test1 = pd.read_csv(self.test_csv_paths['test1'], index_col=0)

        # load GT Counts per class
        with open(self.gt_json_path, 'r') as f:
            #gt_counts_per_class should be a dict of dicts where keys are image filenames and values are dicts of class:count where class is str
            self.gt_counts_per_class = json.load(f)
            # we can also compute total counts per image
            self.total_gt_counts_per_image = {img: sum(class_counts.values()) for img, class_counts in self.gt_counts_per_class.items()}
        
        # load image classes
        self.img_classes = {}
        # img_classes is a dict where keys are image filenames and values are the lists of classes (str) for that image
        with open(self.img_class_txt_path, 'r') as file:
            for line in file:
                line = line.strip().split('\t')
                if len(line) == 2:
                    img_name, classes_str = line
                    classes_list = [cls.strip() for cls in classes_str.split(',')]
                    self.img_classes[img_name] = classes_list
        
        # load split classes
        with open(self.split_classes_file_path, 'r') as f:
            self.split_classes = json.load(f)
        
        # load txt containing images to exclude if provided
        self.imgs_to_exclude = set()
        if self.img_to_exclude_txt:
            with open(self.img_to_exclude_txt, 'r') as f:
                for line in f:
                    self.imgs_to_exclude.add(line.strip())
            
            # remove images to exclude from the dataframes if they exist
            if self.df_test1 is not None and len(self.imgs_to_exclude) > 0:
                self.df_test1 = self.df_test1.drop(self.imgs_to_exclude)
        

    def process_test1_data(self):

        positive_preds = {}
        pos_classes = {}
        max_neg_classes = {}

        all_classes_list = self.df_test1.columns.tolist().copy()

        for img_filename in self.df_test1.index:
            if self.imgs_to_exclude:
                if img_filename in self.imgs_to_exclude:
                    continue

            positive_class_names = self.img_classes[img_filename]

            positive_class_counts = {class_name: self.df_test1.loc[img_filename][class_name] for class_name in positive_class_names}

            positive_preds[img_filename] = positive_class_counts
            
            # set to nan the counts for positive classes
            for class_name in positive_class_names:
                self.df_test1.at[img_filename, class_name] = np.nan
            
            pos_classes[img_filename] = positive_class_names
        

        # compute max negatives among the classes that are not present in each image
        max_negs = self.df_test1.max(axis=1)
        max_neg_classes = self.df_test1.idxmax(axis=1) # this gives the class name with max negative prediction per image
        negative_preds_mean = self.df_test1.mean(axis=1)



        #positive_preds = pd.DataFrame.from_dict(positive_preds, orient='index', fill_value=np.nan)
        # TypeError: from_dict() got an unexpected keyword argument 'fill_value'
        positive_preds = pd.DataFrame.from_dict(positive_preds, orient='index')
        # to apply fill_value, we use fillna
        positive_preds = positive_preds.fillna(np.nan)

        positive_preds.index.name = 'Image Name'
        positive_preds.reset_index(inplace=True)
        # positive_preds will have this structure:
        # Image Name | class1 | class2 | ... | class J | ... | class N
        # img1.jpg | 3 | 5 | ... | nan | ... | nan
        # img2.jpg | nan | nan | ... | 2 | ... | nan
        # where class1 ... classN are all the classes in the dataset, and for each image only the present classes have counts, others are nan

        # now we build gt_counts DataFrame with the same structure
        #gt_counts = pd.DataFrame.from_dict(self.gt_counts_per_class, orient='index', fill_value=np.nan)
        gt_counts = pd.DataFrame.from_dict(self.gt_counts_per_class, orient='index')
        # to apply fill_value, we use fillna
        gt_counts = gt_counts.fillna(np.nan)
        gt_counts.index.name = 'Image Name'
        if self.imgs_to_exclude:
            gt_counts = gt_counts.drop(self.imgs_to_exclude)
        gt_counts.reset_index(inplace=True)

        assert len(positive_preds) == len(self.df_test1.index), "Number of images in positive predictions does not match number of images in test1 dataframe. Values: {} vs {}".format(len(positive_preds), len(self.df_test1.index))
        assert len(negative_preds_mean) == len(self.df_test1.index), "Number of images in negative predictions does not match number of images in test1 dataframe. Values: {} vs {}".format(len(negative_preds_mean), len(self.df_test1.index))

        mean_positive_preds = positive_preds.set_index(['Image Name']).mean(axis=1)

        self.aggregation_df = pd.DataFrame(data={
            'Image Name': self.df_test1.index,
            'Max Negative Pred': max_negs,
            'Max Negative Class': max_neg_classes,
            'Mean Negative Pred': negative_preds_mean,
            'Mean Positive Pred': mean_positive_preds,
            #'Positive Pred': positive_preds.apply(lambda row: {col: row[col] for col in positive_preds.columns if col != 'Image Name' and not pd.isna(row[col])}, axis=1),
            # 'GT Counts': gt_counts.apply(lambda row: {col: row[col] for col in gt_counts.columns if col != 'Image Name' and not pd.isna(row[col])}, axis=1),
        })

        self.aggregation_df = self.aggregation_df \
            .merge(positive_preds, on='Image Name', how='left') \
            .merge(gt_counts, on='Image Name', how='left', suffixes=('_Pred', '_GT'))

        # {class}_positive-gt_gap is defined as the absolute difference between the GT count and the positive prediction of each positive class for each image
        # {class}_negative-gt_gap is defined as the absolute difference between the mean negative pred and the gt count for each positive class for each image
        # {class}_positive-negative_gap is defined as the absolute difference between the positive prediction and the mean negative prediction for each positive class for each image
        # iterate over the classes present in the dataset
        for class_name in all_classes_list:
            self.aggregation_df[f"{class_name}_positive-GT_gap"] = (self.aggregation_df[f"{class_name}_GT"] - self.aggregation_df[f"{class_name}_Pred"]).abs()
            self.aggregation_df[f"{class_name}_negative-GT_gap"] = (self.aggregation_df[f"{class_name}_GT"] - self.aggregation_df["Mean Negative Pred"]).abs()
            self.aggregation_df[f"{class_name}_positive-negative_gap"] = (self.aggregation_df[f"{class_name}_Pred"] - self.aggregation_df["Mean Negative Pred"]).abs()

            # where the gt count is nan, we don't compute the gaps
            self.aggregation_df.loc[self.aggregation_df[f"{class_name}_GT"].isna(), f"{class_name}_positive-GT_gap"] = np.nan
            self.aggregation_df.loc[self.aggregation_df[f"{class_name}_GT"].isna(), f"{class_name}_negative-GT_gap"] = np.nan
            self.aggregation_df.loc[self.aggregation_df[f"{class_name}_GT"].isna(), f"{class_name}_positive-negative_gap"] = np.nan

            # prevent fragmentation issues
            self.aggregation_df = self.aggregation_df.copy()

        # compute overall metrics
        # for each row, we average the positive-negative, positive-gt and negative-gt gaps for the present classes (where the cell is not np.nan)
        def compute_mean_gap(row, gap_type):
            gaps = []
            for class_name in all_classes_list:
                if class_name in pos_classes[row['Image Name']]:
                    gap_value = row[f"{class_name}_{gap_type}"]
                    if not pd.isna(gap_value):
                        gaps.append(gap_value)
            if len(gaps) > 0:
                return np.mean(gaps)
            else:
                return np.nan
        self.aggregation_df['Mean Positive-GT Gap'] = self.aggregation_df.apply(lambda row: compute_mean_gap(row, 'positive-GT_gap'), axis=1)
        self.aggregation_df['Mean Negative-GT Gap'] = self.aggregation_df.apply(lambda row: compute_mean_gap(row, 'negative-GT_gap'), axis=1)
        self.aggregation_df['Mean Positive-Negative Gap'] = self.aggregation_df.apply(lambda row: compute_mean_gap(row, 'positive-negative_gap'), axis=1)

        self.aggregation_df.reset_index(drop=True, inplace=True)

    def evaluate_test1_metrics(self, normalization='average_gt'):
        """
        Evaluate metrics for Test 1 (Multiclass Negative Label Test).
        normalization: str, normalization method for gaps. Options are 'average_gt', 'max_gt', 'total_gt'
        
        normalization factor for gaps can be either:
        - a. 'average_gt' average GT count for present classes for each image
        - b. 'max_gt' max GT count for present classes for each image
        - c. 'total_gt' total GT count for each image
        """

        if normalization not in ['average_gt', 'max_gt', 'total_gt']:
            raise ValueError("Normalization method not recognized. Choose from 'average_gt', 'max_gt', 'total_gt'.")

        
        def compute_normalization_factor(row):
            gt_counts = []
            for class_name in self.img_classes[row['Image Name']]:
                gt_value = row[f"{class_name}_GT"]
                if not pd.isna(gt_value):
                    gt_counts.append(gt_value)
            if len(gt_counts) > 0:
                if normalization == 'max_gt':
                    return np.max(gt_counts)
                elif normalization == 'total_gt':
                    return np.sum(gt_counts)
                else:  # 'average_gt'
                    return np.mean(gt_counts)
            else:
                print(f"Warning: No GT counts found for image {row['Image Name']} when computing normalization factor. Setting to 1.0 to avoid division by zero.")
                return 1.0  # to avoid division by zero
        
        self.aggregation_df['Normalization Factor'] = self.aggregation_df.apply(compute_normalization_factor, axis=1)

        # Now compute final metrics
        self.aggregation_df['Mean Positive Pred Normalized'] = self.aggregation_df['Mean Positive Pred'] / self.aggregation_df['Normalization Factor']
        self.aggregation_df['Mean Negative Pred Normalized'] = self.aggregation_df['Mean Negative Pred'] / self.aggregation_df['Normalization Factor']

        pos_pred_normalized_mean = round(self.aggregation_df['Mean Positive Pred Normalized'].mean(), self.metric_precision)
        neg_pred_normalized_mean = round(self.aggregation_df['Mean Negative Pred Normalized'].mean(), self.metric_precision)


        good_preds = len(self.aggregation_df[self.aggregation_df['Mean Positive-GT Gap'] < self.aggregation_df['Mean Negative-GT Gap']])
        all_preds = len(self.aggregation_df)
        positive_prediction_rate = round((good_preds / all_preds) * 100, 2)

        mae = round(self.aggregation_df["Mean Positive-GT Gap"].mean(), self.metric_precision)
        rmse = round(np.sqrt((self.aggregation_df["Mean Positive-GT Gap"] ** 2).mean()), self.metric_precision)

        return {
            'Model': [self.model_name],
            'AvgNP' : [pos_pred_normalized_mean],
            'AvgNMN' : [neg_pred_normalized_mean],
            'PCCN' : [positive_prediction_rate],
            'MAE' : [mae],
            'RMSE' : [rmse]
        }

            

        





