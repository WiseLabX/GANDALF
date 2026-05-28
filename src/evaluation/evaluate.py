
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Baseline performances
def malwaredetector_performance(X_train, X_test, y_train, y_test,test_malfamily):

    X_test = X_test.reset_index(drop=True)
    test_malfamily = test_malfamily.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    X_train_scaled = X_train
    X_test_scaled = X_test

    # models
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM (RBF Kernel)": SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, class_weight='balanced', random_state=42),
        "MLP Neural Network": MLPClassifier(
            hidden_layer_sizes=(100,),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )
    }

    print("\n=== Malware Detector Performans Comparison ===\n")

    for name, model in classifiers.items():
        print(f"--- {name} ---")
        # model.fit(X_train_scaled, y_train)
        # y_pred = model.predict(X_test_scaled)
        X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_df = pd.DataFrame(X_test_scaled, columns=X_train.columns)

        model.fit(X_train_df, y_train)
        y_pred = model.predict(X_test_df)

        print("Index alignment check baseline:",
              X_test_df.index.equals(test_malfamily.index))

        #
        y_prob = model.predict_proba(X_test_df)[:, 1]

        #
        auc = roc_auc_score(y_test, y_prob)

        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average='macro')
        rec_macro = recall_score(y_test, y_pred, average='macro')
        f1_macro = f1_score(y_test, y_pred, average='macro')

        rec_malware = recall_score(y_test, y_pred, pos_label=1)
        rec_benign = recall_score(y_test, y_pred, pos_label=0)

        prec_malware = precision_score(y_test, y_pred, pos_label=1)
        f1_malware = f1_score(y_test, y_pred, pos_label=1)

        # confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        print(f"Accuracy: {acc:.4f}")
        print(f"Precision (macro): {prec_macro:.4f}")
        print(f"Recall (macro): {rec_macro:.4f}")
        print(f"F1-score (macro): {f1_macro:.4f}")

        print(f"Recall (malware): {rec_malware:.4f}")
        print(f"Precision (malware): {prec_malware:.4f}")
        print(f"F1 (malware): {f1_malware:.4f}")

        print(f"Recall (benign): {rec_benign:.4f}")
        print(f"FPR: {fpr:.4f}")
        print(f"AUC: {auc:.4f}")

        print(classification_report(y_test, y_pred))
        print("--------------------------------------\n")


        results_df = pd.DataFrame({
            'y_true': y_test.values,
            'y_pred': y_pred,
            'MalFamily': test_malfamily.values
        })

        malware_df = results_df[results_df['y_true'] == 1]

        summary = malware_df.groupby('MalFamily').apply(
            lambda x: pd.Series({
                'Total_in_Test': len(x),
                'True_Positive': (x['y_pred'] == 1).sum(),
                'False_Negative': (x['y_pred'] == 0).sum()
            })
        ).reset_index()

        print("Malware Family Based Results:")
        print(summary.sort_values(by='False_Negative', ascending=False))

# Synthetic Data Only performances
def malwaredetector_performance_fakedata(fake_data_binary,X_train, X_test, y_train, y_test):

    X_train_scaled = X_train
    X_test_scaled_fakedata = fake_data_binary

    # models
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=2000, solver='lbfgs', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM (RBF Kernel)": SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, class_weight='balanced', random_state=42),
        "MLP Neural Network": MLPClassifier(
            hidden_layer_sizes=(100,),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )
    }

    print("\n Malware Detector Performans (Fake Data) Comparison \n")

    y_test = np.ones(len(fake_data_binary), dtype=int)

    for name, model in classifiers.items():
        print(f"--- {name} ---")

        X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_df = pd.DataFrame(X_test_scaled_fakedata, columns=X_train.columns)

        model.fit(X_train_df, y_train)
        y_pred = model.predict(X_test_df)

        y_prob = model.predict_proba(X_test_df)[:, 1]

        auc = roc_auc_score(y_test, y_prob)

        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average='macro')
        rec_macro = recall_score(y_test, y_pred, average='macro')
        f1_macro = f1_score(y_test, y_pred, average='macro')

        rec_malware = recall_score(y_test, y_pred, pos_label=1)
        rec_benign = recall_score(y_test, y_pred, pos_label=0)

        prec_malware = precision_score(y_test, y_pred, pos_label=1)
        f1_malware = f1_score(y_test, y_pred, pos_label=1)

        # confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        print(f"Accuracy: {acc:.4f}")
        print(f"Precision (macro): {prec_macro:.4f}")
        print(f"Recall (macro): {rec_macro:.4f}")
        print(f"F1-score (macro): {f1_macro:.4f}")

        print(f"Recall (malware): {rec_malware:.4f}")
        print(f"Precision (malware): {prec_malware:.4f}")
        print(f"F1 (malware): {f1_malware:.4f}")

        print(f"Recall (benign): {rec_benign:.4f}")
        print(f"FPR: {fpr:.4f}")
        print(f"AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("--------------------------------------\n")

# Baseline + WGAN performances
def malwaredetector_performance_modified(fake_data_binary,X_train, X_test, y_train, y_test, test_malfamily):
    X_test = X_test.reset_index(drop=True)
    test_malfamily = test_malfamily.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    aux_classifier = LogisticRegression()

    X_traincombined = np.concatenate([X_train.values, fake_data_binary], axis=0)
    y_traincombined = np.concatenate([y_train.values, np.array([1] * len(fake_data_binary))], axis=0)

    X_train_scaled = X_traincombined
    X_test_scaled = X_test

    # models
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM (RBF Kernel)": SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, class_weight='balanced', random_state=42),
        "MLP Neural Network": MLPClassifier(
            hidden_layer_sizes=(100,),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )
    }

    print("\n Baseline + WGAN Performans Comparison \n")

    for name, model in classifiers.items():
        print(f"--- {name} ---")
        # model.fit(X_train_scaled, y_traincombined)
        # y_pred = model.predict(X_test_scaled)

        X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_df = pd.DataFrame(X_test_scaled, columns=X_train.columns)

        model.fit(X_train_df, y_traincombined)
        y_pred = model.predict(X_test_df)

        print("Index alignment check baseline + GAN:",
              X_test_df.index.equals(test_malfamily.index))

        y_prob = model.predict_proba(X_test_df)[:, 1]

        auc = roc_auc_score(y_test, y_prob)

        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average='macro')
        rec_macro = recall_score(y_test, y_pred, average='macro')
        f1_macro = f1_score(y_test, y_pred, average='macro')

        rec_malware = recall_score(y_test, y_pred, pos_label=1)
        rec_benign = recall_score(y_test, y_pred, pos_label=0)

        prec_malware = precision_score(y_test, y_pred, pos_label=1)
        f1_malware = f1_score(y_test, y_pred, pos_label=1)

        # confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        print(f"Accuracy: {acc:.4f}")
        print(f"Precision (macro): {prec_macro:.4f}")
        print(f"Recall (macro): {rec_macro:.4f}")
        print(f"F1-score (macro): {f1_macro:.4f}")

        print(f"Recall (malware): {rec_malware:.4f}")
        print(f"Precision (malware): {prec_malware:.4f}")
        print(f"F1 (malware): {f1_malware:.4f}")

        print(f"Recall (benign): {rec_benign:.4f}")
        print(f"FPR: {fpr:.4f}")
        print(f"AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("--------------------------------------\n")

        results_df = pd.DataFrame({
            'y_true': y_test.values,
            'y_pred': y_pred,
            'MalFamily': test_malfamily.values
        })

        malware_df = results_df[results_df['y_true'] == 1]

        summary = malware_df.groupby('MalFamily').apply(
            lambda x: pd.Series({
                'Total_in_Test': len(x),
                'True_Positive': (x['y_pred'] == 1).sum(),
                'False_Negative': (x['y_pred'] == 0).sum()
            })
        ).reset_index()

        print("Malware Family Based Results:")
        print(summary.sort_values(by='False_Negative', ascending=False))

# Baseline + WGAN + Active Learning performances
def malwaredetector_performance_modified_AL(fake_data_binary, X_train, X_test, y_train, y_test,test_malfamily):
    X_test = X_test.reset_index(drop=True)
    test_malfamily = test_malfamily.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    if hasattr(fake_data_binary, "detach"):
        fake_data_binary = fake_data_binary.detach().cpu().numpy()
    elif hasattr(fake_data_binary, "numpy") and not isinstance(fake_data_binary, np.ndarray):
        fake_data_binary = fake_data_binary.numpy()

    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)
    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test)

    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM (RBF Kernel)": SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, class_weight='balanced', random_state=42),
        "MLP Neural Network": MLPClassifier(
            hidden_layer_sizes=(100,),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )
    }

    # Active Learning strategies
    AL_strategies = {
        "informative": "select_informative_samples",
        "margin": "margin_sampling",
        "entropy": "entropy_sampling",
        "uncertainty": "uncertainty_sampling",
        "diversity": "diversity_based_sampling"
    }

    print("\n Proposed (Baseline + WGAN + Active Learning) Performans Compariosn \n")

    for clf_name, clf in classifiers.items():
        print(f"\n Classifier: {clf_name} \n")

        for al_name, al_func_name in AL_strategies.items():
            print(f"--- Active Learning: {al_name} ---")

            aux_classifier = LogisticRegression(max_iter=2000, solver='lbfgs', random_state=42)

            try:
                al_func = globals()[al_func_name]

                if al_name == "query_by_committee":
                    fake_selected, fake_labels = al_func(
                        fake_data_binary, list(classifiers.values()),
                        X_train, y_train, n_select=active_learning_synthetic_malware_size
                    )
                else:
                    fake_selected, fake_labels = al_func(
                        fake_data_binary, aux_classifier,
                        X_train, y_train, n_select=active_learning_synthetic_malware_size
                    )

            except KeyError:
                print(f"⚠️ {al_func_name} function is not available.")
                continue
            except Exception as e:
                print(f"⚠️ {al_name} error occurred: {str(e)}")
                continue

            if isinstance(fake_selected, np.ndarray):
                try:
                    fake_selected_df = pd.DataFrame(fake_selected, columns=X_train.columns)
                except Exception:
                    fake_selected_df = pd.DataFrame(fake_selected)
            elif isinstance(fake_selected, pd.DataFrame):
                fake_selected_df = fake_selected
            else:
                fake_selected_df = pd.DataFrame(np.array(fake_selected), columns=X_train.columns)

            fake_labels_arr = np.array([1] * len(fake_selected_df))

            X_traincombined = pd.concat([X_train, fake_selected_df], axis=0).reset_index(drop=True)
            y_traincombined = pd.concat([y_train.reset_index(drop=True),
                                         pd.Series(fake_labels_arr)], axis=0).reset_index(drop=True)

            # Train the model
            try:
                clf.fit(X_traincombined, y_traincombined)
                y_pred = clf.predict(X_test)

                y_prob = clf.predict_proba(X_test)[:, 1]

            except Exception as e:
                print(f"Error: {str(e)}")
                continue

            print(
                "Index alignment (X_test vs test_malfamily) WGAN+AL:",
                X_test.index.equals(test_malfamily.index)
            )

            auc = roc_auc_score(y_test, y_prob)

            acc = accuracy_score(y_test, y_pred)
            prec_macro = precision_score(y_test, y_pred, average='macro')
            rec_macro = recall_score(y_test, y_pred, average='macro')
            f1_macro = f1_score(y_test, y_pred, average='macro')

            rec_malware = recall_score(y_test, y_pred, pos_label=1)
            rec_benign = recall_score(y_test, y_pred, pos_label=0)

            prec_malware = precision_score(y_test, y_pred, pos_label=1)
            f1_malware = f1_score(y_test, y_pred, pos_label=1)

            # confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            print(f"Accuracy: {acc:.4f}")
            print(f"Precision (macro): {prec_macro:.4f}")
            print(f"Recall (macro): {rec_macro:.4f}")
            print(f"F1-score (macro): {f1_macro:.4f}")

            print(f"Recall (malware): {rec_malware:.4f}")
            print(f"Precision (malware): {prec_malware:.4f}")
            print(f"F1 (malware): {f1_malware:.4f}")

            print(f"Recall (benign): {rec_benign:.4f}")
            print(f"FPR: {fpr:.4f}")
            print(f"AUC: {auc:.4f}")
            print("--------------------------------------\n")

            results_df = pd.DataFrame({
                'y_true': y_test.values,
                'y_pred': y_pred,
                'MalFamily': test_malfamily.values
            })

            malware_df = results_df[results_df['y_true'] == 1]

            summary = malware_df.groupby('MalFamily').apply(
                lambda x: pd.Series({
                    'Total_in_Test': len(x),
                    'True_Positive': (x['y_pred'] == 1).sum(),
                    'False_Negative': (x['y_pred'] == 0).sum()
                })
            ).reset_index()

            print("Malware Family Based Resutls:")
            print(summary.sort_values(by='False_Negative', ascending=False))


    print("\nAll Classifiers + Active Learning combinations \n")

