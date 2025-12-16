import os
import json
import pandas as pd
import numpy as np

class MulticlassStatisticsExtractor:
    """
    Statistics extractor for multiclass counting evaluation.
    Implements metrics defined in metrics-definitions.md for evaluating counting models
    in a multiclass scenario with negative-label testing.
    """
    
    def __init__(self, model_name, data_dir, test_csv_dir, test_csv_filenames, 
                 gt_json_filename, img_class_txt, split_classes_file, 
                 precision=2, metric_precision=3, img_to_exclude_txt=None):
        """
        Initialize the multiclass statistics extractor.
        
        Args:
            model_name: Name of the model being evaluated
            data_dir: Directory containing ground truth data
            test_csv_dir: Directory containing test CSV files with predictions
            test_csv_filenames: Dictionary with 'test1' key pointing to CSV filename
            gt_json_filename: JSON file with per-class ground truth counts
            img_class_txt: Text file with image-to-classes mapping
            split_classes_file: JSON file with train/val/test split
            precision: Decimal precision for predictions
            metric_precision: Decimal precision for final metrics
            img_to_exclude_txt: Optional text file with images to exclude
        """
        self.model_name = model_name
        self.data_dir = data_dir
        self.test_csv_paths = {key: os.path.join(test_csv_dir, fname) 
                               for key, fname in test_csv_filenames.items()}
        self.gt_json_path = os.path.join(data_dir, gt_json_filename)
        self.img_class_txt_path = os.path.join(data_dir, img_class_txt)
        self.split_classes_file_path = os.path.join(data_dir, split_classes_file)
        self.precision = precision
        self.metric_precision = metric_precision
        
        # Data structures
        self.df_test1 = None
        self.gt_counts_per_class = None  # Dict[img_name, Dict[class_name, count]]
        self.img_classes = None  # Dict[img_name, List[class_name]]
        self.split_classes = None
        self.all_classes = None  # List of all class names
        self.aggregation_df = None
        
        # Handle images to exclude
        self.img_to_exclude_txt = img_to_exclude_txt
        self.imgs_to_exclude = set()
        
    def load_data(self):
        """Load all required data files."""
        # Load predictions CSV
        self.df_test1 = pd.read_csv(self.test_csv_paths['test1'], index_col=0)
        self.all_classes = self.df_test1.columns.tolist()
        
        # Load per-class ground truth counts
        with open(self.gt_json_path, 'r') as f:
            self.gt_counts_per_class = json.load(f)
        
        # Load image-to-classes mapping
        self.img_classes = {}
        with open(self.img_class_txt_path, 'r') as file:
            for line in file:
                line = line.strip().split('\t')
                if len(line) == 2:
                    img_name, classes_str = line
                    classes_list = [cls.strip() for cls in classes_str.split(',')]
                    self.img_classes[img_name] = classes_list
        
        # Load split classes
        with open(self.split_classes_file_path, 'r') as f:
            self.split_classes = json.load(f)
        
        # Load and apply images to exclude
        if self.img_to_exclude_txt and os.path.exists(self.img_to_exclude_txt):
            with open(self.img_to_exclude_txt, 'r') as f:
                for line in f:
                    img_name = line.strip()
                    if not img_name.endswith('.jpg'):
                        img_name = f"{img_name}.jpg"
                    self.imgs_to_exclude.add(img_name)
            
            # Remove excluded images from dataframe
            self.df_test1 = self.df_test1.drop(list(self.imgs_to_exclude), errors='ignore')
    
    def process_test1_data(self):
        """
        Process the test1 data to create an aggregation dataframe.
        Separates positive class predictions from negative class predictions.
        """
        rows = []
        
        for img_filename in self.df_test1.index:
            if img_filename in self.imgs_to_exclude:
                continue
            
            # Get positive and negative classes for this image
            positive_classes = self.img_classes[img_filename]
            negative_classes = [c for c in self.all_classes if c not in positive_classes]
            
            # Extract predictions for positive and negative classes
            positive_preds = {}
            negative_preds = {}
            gt_counts = {}
            
            for class_name in self.all_classes:
                pred = self.df_test1.loc[img_filename, class_name]
                
                if class_name in positive_classes:
                    positive_preds[class_name] = pred
                    gt_counts[class_name] = self.gt_counts_per_class[img_filename][class_name]
                else:
                    negative_preds[class_name] = pred
            
            # Compute aggregated statistics for this image
            row = {
                'Image Name': img_filename,
                'Num Positive Classes': len(positive_classes),
                'Num Negative Classes': len(negative_classes),
            }
            
            # Store individual class predictions and GT
            for class_name in self.all_classes:
                if class_name in positive_classes:
                    row[f'{class_name}_pred'] = positive_preds[class_name]
                    row[f'{class_name}_gt'] = gt_counts[class_name]
                else:
                    row[f'{class_name}_pred'] = negative_preds[class_name]
                    row[f'{class_name}_gt'] = 0
            
            rows.append(row)
        
        self.aggregation_df = pd.DataFrame(rows)
    
    def evaluate_test1_metrics(self):
        """
        Evaluate all metrics according to metrics-definitions.md.
        
        Returns:
            Dictionary with computed metrics including macro and micro averages
            for MNP, NMN, MAE, RMSE, and PCCN.
        """
        N = len(self.aggregation_df)  # Number of images
        
        # Initialize metric accumulators
        mnp_macro_sum = 0
        mnp_micro_numerator = 0
        mnp_micro_denominator = 0
        
        nmn_macro_sum = 0
        nmn_micro_numerator = 0
        nmn_micro_denominator = 0
        
        mae_macro_sum = 0
        mae_micro_numerator = 0
        mae_micro_denominator = 0
        
        rmse_macro_sum = 0
        rmse_micro_numerator = 0
        rmse_micro_denominator = 0
        
        pccn_oneatatime_count = 0
        pccn_avggt_count = 0
        
        # Process each image
        for idx, row in self.aggregation_df.iterrows():
            img_name = row['Image Name']
            positive_classes = self.img_classes[img_name]
            negative_classes = [c for c in self.all_classes if c not in positive_classes]
            
            # Get positive and negative predictions
            pos_preds = [row[f'{c}_pred'] for c in positive_classes]
            pos_gts = [row[f'{c}_gt'] for c in positive_classes]
            neg_preds = [row[f'{c}_pred'] for c in negative_classes]
            
            num_pos = len(positive_classes)
            num_neg = len(negative_classes)
            
            # === Negative-Label Metrics (on negative classes) ===
            
            # MNP (Mean of Negative Predictions)
            if num_neg > 0:
                mnp_i = np.mean(neg_preds)
                mnp_macro_sum += mnp_i
                mnp_micro_numerator += np.sum(neg_preds)
                mnp_micro_denominator += num_neg
            
            # NMN (Normalized Mean of Negative Predictions)
            if num_neg > 0 and num_pos > 0:
                sum_pos_gt = np.sum(pos_gts)
                if sum_pos_gt > 0:
                    nmn_i = np.mean(neg_preds) / sum_pos_gt
                    nmn_macro_sum += nmn_i
                    nmn_micro_numerator += np.sum(neg_preds)
                    nmn_micro_denominator += num_neg * sum_pos_gt
            
            # === Positive-Class Metrics (on positive classes) ===
            
            # MAE (Mean Absolute Error)
            if num_pos > 0:
                errors = [abs(pred - gt) for pred, gt in zip(pos_preds, pos_gts)]
                mae_i = np.mean(errors)
                mae_macro_sum += mae_i
                mae_micro_numerator += np.sum(errors)
                mae_micro_denominator += num_pos
            
            # RMSE (Root Mean Square Error)
            if num_pos > 0:
                squared_errors = [(pred - gt) ** 2 for pred, gt in zip(pos_preds, pos_gts)]
                rmse_i = np.sqrt(np.mean(squared_errors))
                rmse_macro_sum += rmse_i
                rmse_micro_numerator += np.sum(squared_errors)
                rmse_micro_denominator += num_pos
            
            # PCCN (Positive Class Count Nearness)
            if num_pos > 0 and num_neg > 0:
                # d_pos is the same for both variants
                d_pos_i = np.mean([abs(pred - gt) for pred, gt in zip(pos_preds, pos_gts)])
                
                # PCCN_oneatatime: for each positive GT, compute mean distance to all negative preds
                d_neg_oneatatime_list = []
                for pos_gt in pos_gts:
                    distances_to_negs = [abs(pos_gt - neg_pred) for neg_pred in neg_preds]
                    d_neg_oneatatime_list.append(np.mean(distances_to_negs))
                d_neg_oneatatime = np.mean(d_neg_oneatatime_list)
                
                if d_pos_i < d_neg_oneatatime:
                    pccn_oneatatime_count += 1
                
                # PCCN_avgGT: compute average positive GT, then mean distance to negative preds
                avg_pos_gt = np.mean(pos_gts)
                distances_to_negs = [abs(avg_pos_gt - neg_pred) for neg_pred in neg_preds]
                d_neg_avggt = np.mean(distances_to_negs)
                
                if d_pos_i < d_neg_avggt:
                    pccn_avggt_count += 1
        
        # Compute final metrics
        metrics = {
            'Model': [self.model_name],
            'MNP_macro': [round(mnp_macro_sum / N, self.metric_precision) if N > 0 else 0],
            'MNP_micro': [round(mnp_micro_numerator / mnp_micro_denominator, self.metric_precision) 
                          if mnp_micro_denominator > 0 else 0],
            'NMN_macro': [round(nmn_macro_sum / N, self.metric_precision) if N > 0 else 0],
            'NMN_micro': [round(nmn_micro_numerator / nmn_micro_denominator, self.metric_precision) 
                          if nmn_micro_denominator > 0 else 0],
            'PCCN_oneatatime': [round((pccn_oneatatime_count / N) * 100, self.metric_precision) if N > 0 else 0],
            'PCCN_avgGT': [round((pccn_avggt_count / N) * 100, self.metric_precision) if N > 0 else 0],
            'MAE_macro': [round(mae_macro_sum / N, self.metric_precision) if N > 0 else 0],
            'MAE_micro': [round(mae_micro_numerator / mae_micro_denominator, self.metric_precision) 
                          if mae_micro_denominator > 0 else 0],
            'RMSE_macro': [round(rmse_macro_sum / N, self.metric_precision) if N > 0 else 0],
            'RMSE_micro': [round(np.sqrt(rmse_micro_numerator / rmse_micro_denominator), self.metric_precision) 
                           if rmse_micro_denominator > 0 else 0],
        }
        
        return metrics
    
    def get_aggregation_df(self):
        """Return the aggregation dataframe for further analysis."""
        return self.aggregation_df
    
    def save_aggregation_df(self, output_path):
        """Save the aggregation dataframe to CSV."""
        if self.aggregation_df is not None:
            self.aggregation_df.to_csv(output_path, index=False)
