import os
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, recall_score, precision_score, f1_score


def pick_test_inds(y, test_frac=0.2):
    pos_inds = np.where(y == 1)[0]
    neg_inds = np.where(y == 0)[0]
    n_pos_test = int(len(pos_inds) * test_frac)
    n_neg_test = int(len(neg_inds) * test_frac)
    test_pos_inds = np.random.choice(pos_inds, n_pos_test, replace=False)
    test_neg_inds = np.random.choice(neg_inds, n_neg_test, replace=False)
    test_inds = np.concatenate([test_pos_inds, test_neg_inds])
    return sorted(test_inds)


def print_scores(scores_dict):
    for key in scores_dict.keys():
        print(f"--- {key} ---")
        for metric in scores_dict[key].keys():
            metric_vals = np.array(scores_dict[key][metric])
            if len(metric_vals.shape) == 1:
                print(f"{metric}: {np.mean(metric_vals)} +/- {np.std(metric_vals)}")
            else:
                mean_val = np.mean(metric_vals, axis=0)
                std_val = np.std(metric_vals, axis=0)
                print(f"{metric}: {mean_val} +/- {std_val}")


def main():
    parser = argparse.ArgumentParser(description='Train SVM classifiers on gaze features.')
    parser.add_argument('--raw_data_path', type=str, default='raw_data',
                        help='Path to raw data directory (containing transcriptions.csv)')
    parser.add_argument('--processed_data_path', type=str, default='processed_data',
                        help='Path to processed feature CSVs')
    parser.add_argument('--output_path', type=str, default='.',
                        help='Directory to save ROC curve plots')
    args = parser.parse_args()

    transcriptions = pd.read_csv(os.path.join(args.raw_data_path, 'transcriptions.csv'))

    all_data = None
    existence_gts = []
    context_gts = []
    future_gts = []
    for samp in range(len(transcriptions)):
        comp_gaze_data = pd.read_csv(os.path.join(args.processed_data_path, f'{samp}.csv'))
        if all_data is None:
            all_data = comp_gaze_data
        else:
            all_data = pd.concat([all_data, comp_gaze_data], ignore_index=True)

        existence_gt = transcriptions['L1_existence_flag'][samp]
        context_gt = transcriptions['L2_context_flag'][samp]
        future_gt = transcriptions['L3_future_flag'][samp]
        existence_gts += [existence_gt] * len(comp_gaze_data)
        context_gts += [context_gt] * len(comp_gaze_data)
        future_gts += [future_gt] * len(comp_gaze_data)

    all_data['existence_gt'] = existence_gts
    all_data['context_gt'] = context_gts
    all_data['future_gt'] = future_gts
    existence_y = all_data['existence_gt'].values
    context_y = all_data['context_gt'].values
    future_y = all_data['future_gt'].values

    for i in range(len(existence_y)):
        if future_y[i] == 1:
            context_y[i] = 1
            existence_y[i] = 1
        if context_y[i] == 1:
            existence_y[i] = 1

    # 3-class labels: 0=unaware, 1=present-aware (existence+context), 2=future-aware
    y_3class = existence_y.copy()
    y_3class[future_y == 1] = 2

    features_cols = [col for col in all_data.columns if col not in ['existence_gt', 'context_gt', 'future_gt', 'q_image']]
    X = all_data[features_cols].values

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    print("3-class label distribution:", dict(zip(*np.unique(y_3class, return_counts=True))))

    # ------------------------------------------------------------------ #
    # Binary SVMs (one per SA level)                                       #
    # ------------------------------------------------------------------ #
    binary_scores = {
        'existence': {'accuracy': [], 'uar': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []},
        'context':   {'accuracy': [], 'uar': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []},
        'future':    {'accuracy': [], 'uar': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []},
        'random':    {'accuracy': [], 'uar': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []},
    }

    # ------------------------------------------------------------------ #
    # Cascaded + Hierarchical classifiers (3-class)                        #
    # These share the same trial loop and splits as the binary SVMs.       #
    # ------------------------------------------------------------------ #
    cascaded_scores = {
        'cascaded':    {'accuracy': [], 'uar': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []},
        'hierarchical':{'accuracy': [], 'uar': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []},
    }

    # ------------------------------------------------------------------ #
    # Simple multiclass SVM (3-class, no hierarchical structure)           #
    # ------------------------------------------------------------------ #
    multiclass_scores = {
        'multiclass': {'accuracy': [], 'uar': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []},
        'random':     {'accuracy': [], 'uar': [], 'precision': [], 'recall': [], 'f1': []},
    }

    for num_trial in range(3):
        test_inds = pick_test_inds(existence_y, test_frac=0.2)
        train_inds = np.array([i for i in range(len(existence_y)) if i not in test_inds])

        X_train, X_test = X_scaled[train_inds], X_scaled[test_inds]
        existence_y_train, existence_y_test = existence_y[train_inds], existence_y[test_inds]
        context_y_train, context_y_test = context_y[train_inds], context_y[test_inds]
        future_y_train, future_y_test = future_y[train_inds], future_y[test_inds]
        y_3class_train, y_3class_test = y_3class[train_inds], y_3class[test_inds]

        # --- Existence binary classifier ---
        exist_clf = SVC(kernel="rbf", probability=True, random_state=42, class_weight='balanced')
        exist_clf.fit(X_train, existence_y_train)

        y_pred = exist_clf.predict(X_test)
        accuracy = np.mean(y_pred == existence_y_test)
        recall = recall_score(existence_y_test, y_pred, average=None)
        precision = precision_score(existence_y_test, y_pred, average=None, pos_label=1)
        f1 = f1_score(existence_y_test, y_pred, average=None, pos_label=1)
        uar = np.mean(recall)
        roc_auc = roc_auc_score(existence_y_test, exist_clf.decision_function(X_test))

        fpr, tpr, _ = roc_curve(existence_y_test, exist_clf.decision_function(X_test))
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.savefig(os.path.join(args.output_path, 'existence_roc_curve.png'))
        plt.close()
        print(f"Existence - Accuracy: {accuracy}, UAR: {uar}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, ROC AUC: {roc_auc}")

        binary_scores['existence']['accuracy'].append(accuracy)
        binary_scores['existence']['uar'].append(uar)
        binary_scores['existence']['precision'].append(precision)
        binary_scores['existence']['recall'].append(recall)
        binary_scores['existence']['f1'].append(f1)
        binary_scores['existence']['roc_auc'].append(roc_auc)

        # --- Context binary classifier ---
        context_clf = SVC(kernel="rbf", probability=True, random_state=42, class_weight='balanced')
        context_clf.fit(X_train, context_y_train)

        y_pred = context_clf.predict(X_test)
        accuracy = np.mean(y_pred == context_y_test)
        recall = recall_score(context_y_test, y_pred, average=None)
        precision = precision_score(context_y_test, y_pred, average=None, pos_label=1)
        f1 = f1_score(context_y_test, y_pred, average=None, pos_label=1)
        uar = np.mean(recall)
        roc_auc = roc_auc_score(context_y_test, context_clf.decision_function(X_test))

        fpr, tpr, _ = roc_curve(context_y_test, context_clf.decision_function(X_test))
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.savefig(os.path.join(args.output_path, 'context_roc_curve.png'))
        plt.close()
        print(f"Context - Accuracy: {accuracy}, UAR: {uar}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, ROC AUC: {roc_auc}")

        binary_scores['context']['accuracy'].append(accuracy)
        binary_scores['context']['uar'].append(uar)
        binary_scores['context']['precision'].append(precision)
        binary_scores['context']['recall'].append(recall)
        binary_scores['context']['f1'].append(f1)
        binary_scores['context']['roc_auc'].append(roc_auc)

        # --- Future binary classifier ---
        future_clf = SVC(kernel="rbf", probability=True, random_state=42, class_weight='balanced')
        future_clf.fit(X_train, future_y_train)

        y_pred = future_clf.predict(X_test)
        accuracy = np.mean(y_pred == future_y_test)
        recall = recall_score(future_y_test, y_pred, average=None)
        precision = precision_score(future_y_test, y_pred, average=None, pos_label=1)
        f1 = f1_score(future_y_test, y_pred, average=None, pos_label=1)
        uar = np.mean(recall)
        roc_auc = roc_auc_score(future_y_test, future_clf.decision_function(X_test))

        fpr, tpr, _ = roc_curve(future_y_test, future_clf.decision_function(X_test))
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.savefig(os.path.join(args.output_path, 'future_roc_curve.png'))
        plt.close()
        print(f"Future - Accuracy: {accuracy}, UAR: {uar}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, ROC AUC: {roc_auc}")

        binary_scores['future']['accuracy'].append(accuracy)
        binary_scores['future']['uar'].append(uar)
        binary_scores['future']['precision'].append(precision)
        binary_scores['future']['recall'].append(recall)
        binary_scores['future']['f1'].append(f1)
        binary_scores['future']['roc_auc'].append(roc_auc)

        # --- Random binary baseline ---
        y_pred = np.random.choice([0, 1], size=len(existence_y_test))
        accuracy = np.mean(y_pred == existence_y_test)
        recall = recall_score(existence_y_test, y_pred, average=None)
        precision = precision_score(existence_y_test, y_pred, average=None, pos_label=1)
        f1 = f1_score(existence_y_test, y_pred, average=None, pos_label=1)
        uar = np.mean(recall)
        roc_auc = roc_auc_score(existence_y_test, exist_clf.decision_function(X_test))

        fpr, tpr, _ = roc_curve(existence_y_test, exist_clf.decision_function(X_test))
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.savefig(os.path.join(args.output_path, 'random_roc_curve.png'))
        plt.close()
        print(f"Random - Accuracy: {accuracy}, UAR: {uar}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, ROC AUC: {roc_auc}")

        binary_scores['random']['accuracy'].append(accuracy)
        binary_scores['random']['uar'].append(uar)
        binary_scores['random']['precision'].append(precision)
        binary_scores['random']['recall'].append(recall)
        binary_scores['random']['f1'].append(f1)
        binary_scores['random']['roc_auc'].append(roc_auc)

        # ------------------------------------------------------------------ #
        # Cascaded binary classifiers                                          #
        # Reuses exist_clf and future_clf trained above.                       #
        # p1 = P(present-aware), p2 = P(future-aware)                         #
        # P(unaware)       = 1 - p1                                            #
        # P(present-aware) = p1 * (1 - p2)                                    #
        # P(future-aware)  = p1 * p2                                           #
        # ------------------------------------------------------------------ #
        p1 = exist_clf.predict_proba(X_test)[:, 1]
        p2 = future_clf.predict_proba(X_test)[:, 1]
        joint_probs = np.column_stack([1 - p1, p1 * (1 - p2), p1 * p2])
        y_pred = np.argmax(joint_probs, axis=1)

        accuracy = np.mean(y_pred == y_3class_test)
        recall = recall_score(y_3class_test, y_pred, average=None)
        precision = precision_score(y_3class_test, y_pred, average=None)
        f1 = f1_score(y_3class_test, y_pred, average=None)
        uar = np.mean(recall)
        roc_auc = roc_auc_score(y_3class_test, joint_probs, multi_class='ovr')
        print(f"Cascaded - Accuracy: {accuracy}, UAR: {uar}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, ROC AUC: {roc_auc}")
        print("Confusion Matrix:\n", confusion_matrix(y_3class_test, y_pred))

        cascaded_scores['cascaded']['accuracy'].append(accuracy)
        cascaded_scores['cascaded']['uar'].append(uar)
        cascaded_scores['cascaded']['precision'].append(precision)
        cascaded_scores['cascaded']['recall'].append(recall)
        cascaded_scores['cascaded']['f1'].append(f1)
        cascaded_scores['cascaded']['roc_auc'].append(roc_auc)

        # ------------------------------------------------------------------ #
        # Hierarchical binary classifiers                                      #
        # Each level's predicted probability is appended as an input feature  #
        # to the next level's classifier.                                      #
        # Level 1: exist_clf (reused from above)                               #
        # Level 2: context_hier trained on [X | exist_proba]                  #
        # Level 3: future_hier  trained on [X | exist_proba | context_proba]  #
        # Final 3-class joint probabilities use the same cascaded formula,    #
        # with p1 from context_hier and p2 from future_hier.                  #
        # ------------------------------------------------------------------ #
        exist_proba_train = exist_clf.predict_proba(X_train)[:, 1:]
        exist_proba_test = exist_clf.predict_proba(X_test)[:, 1:]

        X_hier1_train = np.hstack([X_train, exist_proba_train])
        X_hier1_test = np.hstack([X_test, exist_proba_test])
        context_hier = SVC(kernel="rbf", probability=True, random_state=42, class_weight='balanced')
        context_hier.fit(X_hier1_train, context_y_train)

        context_proba_train = context_hier.predict_proba(X_hier1_train)[:, 1:]
        context_proba_test = context_hier.predict_proba(X_hier1_test)[:, 1:]

        X_hier2_train = np.hstack([X_hier1_train, context_proba_train])
        X_hier2_test = np.hstack([X_hier1_test, context_proba_test])
        future_hier = SVC(kernel="rbf", probability=True, random_state=42, class_weight='balanced')
        future_hier.fit(X_hier2_train, future_y_train)

        p1_hier = context_hier.predict_proba(X_hier1_test)[:, 1]
        p2_hier = future_hier.predict_proba(X_hier2_test)[:, 1]
        joint_probs_hier = np.column_stack([1 - p1_hier, p1_hier * (1 - p2_hier), p1_hier * p2_hier])
        y_pred = np.argmax(joint_probs_hier, axis=1)

        accuracy = np.mean(y_pred == y_3class_test)
        recall = recall_score(y_3class_test, y_pred, average=None)
        precision = precision_score(y_3class_test, y_pred, average=None)
        f1 = f1_score(y_3class_test, y_pred, average=None)
        uar = np.mean(recall)
        roc_auc = roc_auc_score(y_3class_test, joint_probs_hier, multi_class='ovr')
        print(f"Hierarchical - Accuracy: {accuracy}, UAR: {uar}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, ROC AUC: {roc_auc}")
        print("Confusion Matrix:\n", confusion_matrix(y_3class_test, y_pred))

        cascaded_scores['hierarchical']['accuracy'].append(accuracy)
        cascaded_scores['hierarchical']['uar'].append(uar)
        cascaded_scores['hierarchical']['precision'].append(precision)
        cascaded_scores['hierarchical']['recall'].append(recall)
        cascaded_scores['hierarchical']['f1'].append(f1)
        cascaded_scores['hierarchical']['roc_auc'].append(roc_auc)

        # ------------------------------------------------------------------ #
        # Simple multiclass SVM (3-class, no hierarchical structure)           #
        # ------------------------------------------------------------------ #
        multi_clf = SVC(kernel="rbf", probability=True, random_state=42, class_weight='balanced')
        multi_clf.fit(X_train, y_3class_train)

        y_pred = multi_clf.predict(X_test)
        accuracy = np.mean(y_pred == y_3class_test)
        recall = recall_score(y_3class_test, y_pred, average=None)
        precision = precision_score(y_3class_test, y_pred, average=None)
        f1 = f1_score(y_3class_test, y_pred, average=None)
        uar = np.mean(recall)
        y_proba = multi_clf.predict_proba(X_test)
        roc_auc = roc_auc_score(y_3class_test, y_proba, multi_class='ovr')
        print(f"Multiclass - Accuracy: {accuracy}, UAR: {uar}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, ROC AUC: {roc_auc}")
        print("Confusion Matrix:\n", confusion_matrix(y_3class_test, y_pred))

        multiclass_scores['multiclass']['accuracy'].append(accuracy)
        multiclass_scores['multiclass']['uar'].append(uar)
        multiclass_scores['multiclass']['precision'].append(precision)
        multiclass_scores['multiclass']['recall'].append(recall)
        multiclass_scores['multiclass']['f1'].append(f1)
        multiclass_scores['multiclass']['roc_auc'].append(roc_auc)

        # Random 3-class baseline
        y_pred = np.random.choice(np.unique(y_3class), size=len(y_3class_test))
        accuracy = np.mean(y_pred == y_3class_test)
        recall = recall_score(y_3class_test, y_pred, average=None)
        precision = precision_score(y_3class_test, y_pred, average=None)
        f1 = f1_score(y_3class_test, y_pred, average=None)
        uar = np.mean(recall)
        print(f"Random (3-class) - Accuracy: {accuracy}, UAR: {uar}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

        multiclass_scores['random']['accuracy'].append(accuracy)
        multiclass_scores['random']['uar'].append(uar)
        multiclass_scores['random']['precision'].append(precision)
        multiclass_scores['random']['recall'].append(recall)
        multiclass_scores['random']['f1'].append(f1)

    print("\n=== Final Binary SVM Scores (3 trials) ===")
    print_scores(binary_scores)

    print("\n=== Final Cascaded / Hierarchical Scores (3 trials) ===")
    print_scores(cascaded_scores)

    print("\n=== Final Multiclass SVM Scores (3 trials) ===")
    print_scores(multiclass_scores)


if __name__ == '__main__':
    main()
