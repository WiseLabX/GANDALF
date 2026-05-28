

import numpy as np
from scipy.stats import entropy
from sklearn.cluster import KMeans

#  Active Learning functions
def select_informative_samples(fake_data_binary, aux_classifier, X_train, y_train, n_select=active_learning_synthetic_malware_size):

    aux_classifier.fit(X_train, y_train)

    probs = aux_classifier.predict_proba(fake_data_binary)

    uncertainty_scores = -np.max(probs, axis=1)

    selected_indices = np.argsort(uncertainty_scores)[-n_select:]

    selected_fake = fake_data_binary[selected_indices]
    selected_labels = np.ones(n_select, dtype=int)

    return selected_fake, selected_labels

def margin_sampling(fake_data_binary, aux_classifier, X_train, y_train, n_select=active_learning_synthetic_malware_size):

    aux_classifier.fit(X_train, y_train)

    probs = aux_classifier.predict_proba(fake_data_binary)

    # Margin score
    margin_scores = np.abs(probs[:, 0] - probs[:, 1])

    # choose the smallest margin
    selected_indices = np.argsort(margin_scores)[:n_select]

    selected_fake = fake_data_binary[selected_indices]
    selected_labels = np.ones(n_select, dtype=int)

    return selected_fake, selected_labels

from scipy.stats import entropy
def entropy_sampling(fake_data_binary, aux_classifier, X_train, y_train, n_select=active_learning_synthetic_malware_size):

    aux_classifier.fit(X_train, y_train)

    probs = aux_classifier.predict_proba(fake_data_binary)

    entropies = np.array([entropy(prob) for prob in probs])

    # choose the highest entropy
    selected_indices = np.argsort(entropies)[-n_select:]

    selected_fake = fake_data_binary[selected_indices]
    selected_labels = np.ones(n_select, dtype=int)

    return selected_fake, selected_labels


def uncertainty_sampling(fake_data_binary, aux_classifier, X_train, y_train, n_select=active_learning_synthetic_malware_size):
    aux_classifier.fit(X_train, y_train)

    probs = aux_classifier.predict_proba(fake_data_binary)

    uncertainty_scores = -np.max(probs, axis=1)

    selected_indices = np.argsort(uncertainty_scores)[-n_select:]

    selected_fake = fake_data_binary[selected_indices]
    selected_labels = np.ones(n_select, dtype=int)

    return selected_fake, selected_labels

from sklearn.metrics import pairwise_distances

def diversity_based_sampling(fake_data_binary, aux_classifier, X_train, y_train, n_select=active_learning_synthetic_malware_size):

    aux_classifier.fit(X_train, y_train)

    distances = pairwise_distances(fake_data_binary)

    diversity_scores = np.sum(distances, axis=1)

    selected_indices = np.argsort(diversity_scores)[:n_select]

    selected_fake = fake_data_binary[selected_indices]
    selected_labels = np.ones(n_select, dtype=int)

    return selected_fake, selected_labels

