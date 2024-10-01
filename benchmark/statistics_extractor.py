import os
import json
import pandas as pd
import numpy as np

class StatisticsExtractor:
    def __init__(self, model_name, data_dir, test_csv_dir, test_csv_filenames, gt_json_filename, img_class_txt, split_classes_file, precision=2, metric_precision=3, img_to_exclude_txt=None):
        
        self.model_name = model_name
        self.data_dir = data_dir
        self.test_csv_paths = {key: os.path.join(test_csv_dir, fname) for key, fname in test_csv_filenames.items()}
        self.gt_json_path = os.path.join(data_dir, gt_json_filename)
        self.img_class_txt_path = os.path.join(data_dir, img_class_txt)
        self.split_classes_file_path = os.path.join(data_dir, split_classes_file)
        self.precision = precision
        self.metric_precision = metric_precision
        self.aggregation_df = None
        self.df_test1 = None
        self.df_upper_test2 = None
        self.df_lower_test2 = None
        self.gt_counts_dict = None
        self.img_class = {}
        self.split_classes = {}
        self.img_to_exclude_txt = os.path.join(data_dir, img_to_exclude_txt) if img_to_exclude_txt is not None else None
        self.imgs_to_exclude =  None

    def load_data(self):
        # Load Test1 CSV file
        self.df_test1 = pd.read_csv(self.test_csv_paths['test1'], index_col=0)
        
        # Load Test2 Upper and Lower CSV files
        self.df_upper_test2 = pd.read_csv(self.test_csv_paths['upper_test2'], index_col=0)
        self.df_lower_test2 = pd.read_csv(self.test_csv_paths['lower_test2'], index_col=0)

        # Load GT Counts JSON
        with open(self.gt_json_path, 'r') as file:
            gt_counts_dict = json.load(file)
            self.gt_counts_dict = {k: len(v['points']) for k, v in gt_counts_dict.items()}

        # Load Image Class .txt
        with open(self.img_class_txt_path, 'r') as file:
            for line in file:
                line = line.strip().split('\t')
                if len(line) == 2:
                    img_name, label = line
                    # img_id = img_name.split(".")[0]
                    self.img_class[img_name] = label

        # Load Split Classes JSON
        with open(self.split_classes_file_path, 'r') as file:
            self.split_classes = json.load(file)

        # Load txt containing images to be excluded
        if self.img_to_exclude_txt:
            with open(self.img_to_exclude_txt) as file:
                self.imgs_to_exclude = [line.rstrip() for line in file]

            # remove images to exclude from the dataframes (the images are in the row index)
            self.df_test1 = self.df_test1.drop(self.imgs_to_exclude, errors='ignore')
            self.df_upper_test2 = self.df_upper_test2.drop(self.imgs_to_exclude, errors='ignore')
            self.df_lower_test2 = self.df_lower_test2.drop(self.imgs_to_exclude, errors='ignore')

    def process_test1_data(self):
        # gt_counts = list(self.gt_counts_dict.values())

        positive_preds = {}
        pos_classes = {}
        max_neg_classes = {}

        for img_filename in self.df_test1.index:
            if self.imgs_to_exclude:
                if img_filename in self.imgs_to_exclude:
                    continue

            class_name = self.img_class[img_filename]
            
            positive_preds[img_filename] = self.df_test1.loc[img_filename][class_name]
            self.df_test1.loc[img_filename][class_name] = np.nan
            pos_classes[img_filename] = class_name
            
        max_negs = self.df_test1.max(axis=1)
        max_neg_classes = self.df_test1.idxmax(axis=1)
        negative_preds_mean = self.df_test1.mean(axis=1)

        # positive_preds = [np.round(x, self.precision) for x in list(positive_preds.values())]
        positive_preds = pd.DataFrame.from_dict(positive_preds, orient='index', columns=['Positive Pred'])
        positive_preds.index.name = 'Image Name'
        positive_preds.reset_index(inplace=True)
       #  negative_preds_mean = neg_pred_count_means.values.round(self.precision)

        gt_counts = pd.DataFrame.from_dict(self.gt_counts_dict, orient='index', columns=['GT Count'])
        gt_counts.index.name = 'Image Name'
        gt_counts.drop(self.imgs_to_exclude, errors='ignore', inplace=True)
        gt_counts.reset_index(inplace=True)
        pos_classes = pd.DataFrame.from_dict(pos_classes, orient='index', columns=['Positive Class'])
        pos_classes.index.name = 'Image Name'
        pos_classes.reset_index(inplace=True)

        assert len(positive_preds) == len(self.df_test1.index)
        assert len(negative_preds_mean) == len(self.df_test1.index)
        # assert len(gt_counts) == len(self.df_test1.index)

        self.aggregation_df = pd.DataFrame(data={
            'Image Name': self.df_test1.index,
            # 'Positive Class': list(pos_classes.values()),
            # 'GT Count': gt_counts,
            # 'Positive Pred': positive_preds,
            'Mean Negative Pred': negative_preds_mean,
            'Max Neg Class': max_neg_classes,
            'Max Neg Pred': max_negs,
            # 'Positive-Negative Gap': abs(np.array(positive_preds) - negative_preds_mean),
            # "Positive-GT Gap": abs(np.array(gt_counts) - np.array(positive_preds)),
            # "Negative-GT Gap": abs(np.array(gt_counts) - np.array(negative_preds_mean))
        })

        self.aggregation_df = self.aggregation_df \
            .merge(gt_counts, on='Image Name', how='left') \
            .merge(positive_preds, on='Image Name', how='left') \
            .merge(pos_classes, on='Image Name', how='left')
        
        self.aggregation_df["Positive-GT Gap"] = (self.aggregation_df["GT Count"] - self.aggregation_df["Positive Pred"]).abs()
        self.aggregation_df["Negative-GT Gap"] = (self.aggregation_df["GT Count"] - self.aggregation_df["Mean Negative Pred"]).abs()
        self.aggregation_df["Positive-Negative Gap"] = (self.aggregation_df["Positive Pred"] - self.aggregation_df["Mean Negative Pred"]).abs()

        self.aggregation_df.reset_index(drop=True, inplace=True)

    def process_test2_data(self):
        image_names = self.df_upper_test2.index

        # self.df_upper_test2 = self.df_upper_test2.reset_index(drop=True)
        # self.df_lower_test2 = self.df_lower_test2.reset_index(drop=True)

        # Set to 0 all negative predictions
        self.df_upper_test2[self.df_upper_test2 < 0] = 0.0
        self.df_lower_test2[self.df_lower_test2 < 0] = 0.0

        pos_classes = self.df_upper_test2.apply(lambda row: row.index[row.isna()][0] if any(row.isna()) else None, axis=1)

        positive_img_preds_mean = self.df_upper_test2.mean(axis=1)
        negative_img_preds_mean = self.df_lower_test2.mean(axis=1)

        gt_df = pd.DataFrame(index=self.df_upper_test2.index, columns=self.df_upper_test2.columns)
        for k in self.df_lower_test2.index:
            value = self.gt_counts_dict[k]
            gt_df.loc[k, :] = value

        # Obtain Recall dataframe
        gt_df_no_nan = gt_df.replace(0, np.nan)
        min_upper_test_and_gt = self.df_upper_test2.where(self.df_upper_test2 < gt_df_no_nan, gt_df_no_nan)
        recall_df = min_upper_test_and_gt.div(gt_df.replace(0, np.nan))
        # recall_df = recall_df.clip(upper=1.0)

        # Obtain exceedings positive predictions in positive image
        # exceedings_df = self.df_upper_test2 - gt_df
        # exceedings_df[exceedings_df < 0] = 0.0

        # Obtain Precision dataframe
        precision_df = min_upper_test_and_gt.div((self.df_upper_test2 + self.df_lower_test2).replace(0, np.nan))

        # Obtain F-score dataframe 
        fscore_df = (2 * (precision_df * recall_df)).div((precision_df + recall_df).replace(0, np.nan))

        recall_per_row = recall_df.mean(axis=1)
        precision_per_row = precision_df.mean(axis=1)
        fscore_per_row = fscore_df.mean(axis=1)

        self.aggregation_df = pd.DataFrame({'Image Name': image_names,
                                            'Pos Class' : pos_classes,
                                            # 'GT Count': gt_counts,
                                            'Mean Positive Pred': np.array(positive_img_preds_mean.values),
                                            'Mean Negative Pred': np.array(negative_img_preds_mean.values),
                                            'Recall': np.array(recall_per_row.values),
                                            'Precision': np.array(precision_per_row.values),
                                            'F-score': np.array(fscore_per_row.values)
                                            })

    def evaluate_test1_metrics(self):
        self.aggregation_df["Positive Pred Normalized by GT"] = self.aggregation_df["Positive Pred"] / self.aggregation_df["GT Count"]
        self.aggregation_df["Mean Negative Pred Normalized by GT"] = self.aggregation_df["Mean Negative Pred"] / self.aggregation_df["GT Count"]

        pos_pred_normalized_mean = round(self.aggregation_df["Positive Pred Normalized by GT"].mean(), self.metric_precision)
        neg_pred_normalized_mean = round(self.aggregation_df["Mean Negative Pred Normalized by GT"].mean(), self.metric_precision)

        good_preds = len(self.aggregation_df[self.aggregation_df['Positive-GT Gap'] < self.aggregation_df['Negative-GT Gap']])
        all_preds = len(self.aggregation_df)
        positive_prediction_rate = round((good_preds/all_preds)*100, 2)

        mae = round(self.aggregation_df["Positive-GT Gap"].mean(), self.metric_precision)
        rmse = round(np.sqrt((self.aggregation_df["Positive-GT Gap"]**2).mean()), self.metric_precision)

        return {
            'Model': [self.model_name],
            'AvgNP': [pos_pred_normalized_mean],
            'AvgNMN': [neg_pred_normalized_mean],
            'PCCN': [positive_prediction_rate],
            'MAE': [mae],
            'RMSE': [rmse]
        }

    def evaluate_test2_metrics(self):
        recall_mean = round(self.aggregation_df['Recall'].mean(), self.metric_precision)
        precision_mean = round(self.aggregation_df['Precision'].mean(), self.metric_precision)
        fscore_mean = round(self.aggregation_df['F-score'].mean(), self.metric_precision)

        return {
            'Model': [self.model_name],
            'AvgCntRecall': [recall_mean],
            'AvgCntPrecision': [precision_mean],
            'AvgCntFscore': [fscore_mean],
        }

    def save_statistics(self, statistics_data, global_csv_path):
        df_statistics = pd.DataFrame(statistics_data)

        # if os.path.exists(global_csv_path):
        #     df_global = pd.read_csv(global_csv_path)
        # else:
        #     df_global = pd.DataFrame()

        # df_global = pd.concat([df_global, df_statistics], ignore_index=True)
        df_statistics.to_csv(global_csv_path, index=False)
