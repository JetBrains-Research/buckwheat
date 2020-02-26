"""
Topic modeling related functionality.
"""

from typing import Tuple

import artm
import os
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt


def create_batches(directory: str, name: str) -> Tuple[artm.BatchVectorizer, artm.Dictionary]:
    """
    Create the batches and the dictionary from the dataset.
    :param directory: the directory with the dataset.
    :param name: name of the processed dataset.
    :return: BatchVectorizer and Dictionary.
    """
    batch_vectorizer = artm.BatchVectorizer(data_path=directory, data_format='bow_uci',
                                            collection_name=name,
                                            target_folder=os.path.abspath(os.path.join(directory, name + '_batches')))
    dictionary = batch_vectorizer.dictionary
    return batch_vectorizer, dictionary


def define_model(number_of_topics: int, dictionary: artm.Dictionary, sparce_theta: float, sparse_phi: float,
                 decorrelator_phi: float) -> artm.artm_model.ARTM:
    """
    Define the ARTM model.
    :param number_of_topics: number of topics.
    :param dictionary: Batch Vectorizer dictionary.
    :param sparce_theta: Sparse Theta Parameter.
    :param sparse_phi: Sparse Phi Parameter.
    :param decorrelator_phi: Decorellator Phi Parameter.
    :return: ARTM model.
    """
    topic_names = ['topic_{}'.format(i) for i in range(1, number_of_topics + 1)]
    model_artm = artm.ARTM(topic_names=topic_names, cache_theta=True,
                           scores=[artm.PerplexityScore(name='PerplexityScore',
                                                        dictionary=dictionary),
                                   artm.SparsityPhiScore(name='SparsityPhiScore'),
                                   artm.SparsityThetaScore(name='SparsityThetaScore'),
                                   artm.TopicKernelScore(name='TopicKernelScore',
                                                         probability_mass_threshold=0.3),
                                   artm.TopTokensScore(name='TopTokensScore', num_tokens=15)],
                           regularizers=[artm.SmoothSparseThetaRegularizer(name='SparseTheta',
                                                                           tau=sparce_theta),
                                         artm.SmoothSparsePhiRegularizer(name='SparsePhi', tau=sparse_phi),
                                         artm.DecorrelatorPhiRegularizer(name='DecorrelatorPhi', tau=decorrelator_phi)])
    return model_artm


def train_model(model: artm.artm_model.ARTM, number_of_document_passes: int, number_of_collection_passes: int,
                dictionary: artm.Dictionary, batch_vectorizer: artm.BatchVectorizer) -> None:
    """
    Train the ARTM model.
    :param model: the trained model.
    :param number_of_document_passes: number of document passes.
    :param number_of_collection_passes: number of collection passes.
    :param dictionary: Batch Vectorizer dictionary.
    :param batch_vectorizer: Batch Vectorizer.
    :return: None.
    """
    model.num_document_passes = number_of_document_passes
    model.initialize(dictionary=dictionary)
    model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=number_of_collection_passes)


def save_parameters(model: artm.artm_model.ARTM, directory: str, name: str) -> None:
    """
    Save the parameters of the model.
    :param model: the model.
    :param directory: the directory with the dataset.
    :param name: name of the processed dataset.
    :return: None.
    """
    with open(os.path.abspath(os.path.join(directory, 'results', name + '_parameters.txt')), 'w+') as fout:
        fout.write('Sparsity Phi: {0:.3f}'.format(
            model.score_tracker['SparsityPhiScore'].last_value) + '\n')
        fout.write('Sparsity Theta: {0:.3f}'.format(
            model.score_tracker['SparsityThetaScore'].last_value) + '\n')
        fout.write('Kernel contrast: {0:.3f}'.format(
            model.score_tracker['TopicKernelScore'].last_average_contrast) + '\n')
        fout.write('Kernel purity: {0:.3f}'.format(
            model.score_tracker['TopicKernelScore'].last_average_purity) + '\n')
        fout.write('Perplexity: {0:.3f}'.format(
            model.score_tracker['PerplexityScore'].last_value) + '\n')

    plt.plot(range(model.num_phi_updates),
             model.score_tracker['PerplexityScore'].value, 'r--', linewidth=2)
    plt.xlabel('Iterations count')
    plt.ylabel('Perplexity')
    plt.grid(True)
    plt.savefig(os.path.abspath(os.path.join(directory, 'results', name + '_perplexity.png')))
    plt.close()

    plt.plot(range(model.num_phi_updates),
             model.score_tracker['SparsityPhiScore'].value, 'r--', linewidth=2)
    plt.xlabel('Iterations count')
    plt.ylabel('Phi Sparsity')
    plt.grid(True)
    plt.savefig(os.path.abspath(os.path.join(directory, 'results', name + '_phi_sparsity.png')))
    plt.close()

    plt.plot(range(model.num_phi_updates),
             model.score_tracker['SparsityThetaScore'].value, 'r--', linewidth=2)
    plt.xlabel('Iterations count')
    plt.ylabel('Theta Sparsity')
    plt.grid(True)
    plt.savefig(os.path.abspath(os.path.join(directory, 'results', name + '_theta_sparsity.png')))
    plt.close()


def save_most_popular_tokens(model: artm.artm_model.ARTM, directory: str, name: str) -> None:
    """
    Save the most popular tokens of the model.
    :param model: the model.
    :param directory: the directory with the dataset.
    :param name: name of the processed dataset.
    :return: None.
    """
    with open(os.path.abspath(os.path.join(directory, 'results', name + '_most_popular_tokens.txt')), 'w+') as fout:
        for topic_name in model.topic_names:
            fout.write(topic_name + ' : ' + str(model.score_tracker['TopTokensScore'].last_tokens[topic_name]) + '\n')


def save_matrices(model: artm.artm_model.ARTM, directory: str, name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Save the Phi and Theta matrices.
    :param model: the model.
    :param directory: the directory with the dataset.
    :param name: name of the processed dataset.
    :return: Two matrices as DataFrames.
    """
    phi_matrix = model.get_phi().sort_index(axis=0)
    phi_matrix.to_csv(os.path.abspath(os.path.join(directory, 'results', name + '_phi.csv')))
    theta_matrix = model.get_theta().sort_index(axis=1)
    theta_matrix.to_csv(os.path.abspath(os.path.join(directory, 'results', name + '_theta.csv')))
    return phi_matrix, theta_matrix


def save_most_topical_files(number_of_topics: int, theta: pd.DataFrame, directory: str, name: str) -> None:
    """
    Save the most topical files of the model.
    :param number_of_topics: the number of topics.
    :param theta: Theta matrix.
    :param directory: the directory with the dataset.
    :param name: name of the processed dataset.
    :return: None.
    """
    file_address = {}
    with open(os.path.abspath(os.path.join(directory, name + '_tokens.txt')), 'r') as fin:
        for line in fin:
            file_address[int(line.split(';')[0])] = line.split(';')[1]
    with open(os.path.abspath(os.path.join(directory, 'results', name + '_most_topical_files.txt')), 'w+') as fout:
        for i in range(1, number_of_topics + 1):
            fout.write('Topic ' + str(i) + '\n\n')
            dictionary_of_the_topic = theta.sort_values(by='topic_' + str(i), axis=1,
                                                        ascending=False).loc['topic_' + str(i)][:10].to_dict()
            for j in dictionary_of_the_topic.keys():
                fout.write(str(j) + ';' + str(dictionary_of_the_topic[j]) + ';' + file_address[int(j)] + '\n')
            print('\n')


def save_dynamics(directory: str, name: str) -> None:
    """
    Save figures with the dynamics.
    :param directory: the directory with the dataset.
    :param name: name of the processed dataset.
    :return: None.
    """
    indexes = {}
    with open(os.path.abspath(os.path.join(directory, name + '_tokens_info.txt')), 'r') as fin:
        for line in fin:
            indexes[line.rstrip().split(';')[0]] = (int(line.rstrip().split(';')[1].split(',')[0]),
                                                    int(line.rstrip().split(';')[1].split(',')[1]))
    topics_weight = []
    with open(os.path.abspath(os.path.join(directory, 'results', name + '_theta.csv')), 'r') as fin:
        reader = csv.reader(fin)
        next(reader, None)
        for row in reader:
            topics_weight.append([])
            for year in indexes.keys():
                topics_weight[-1].append(sum(float(i) for i in row[indexes[year][0]:indexes[year][1] + 1]))
    topics_weight = np.asarray(topics_weight)
    topics_weight_percent = np.zeros((topics_weight.shape[0], topics_weight.shape[1]))
    for i in range(topics_weight.shape[0]):
        for j in range(topics_weight.shape[1]):
            topics_weight_percent[i, j] = topics_weight[i, j] / np.sum(topics_weight[:, j], keepdims=True) * 100
    np.savetxt(os.path.abspath(os.path.join(directory, 'results', name + '_dynamics.txt')), topics_weight, '%10.5f')
    np.savetxt(os.path.abspath(os.path.join(directory, 'results', name + '_dynamics_percent.txt')),
               topics_weight_percent, '%10.5f')

    plt.stackplot(indexes.keys(), topics_weight)
    plt.xlabel('Year')
    plt.ylabel('Proportion (a. u.)')
    plt.savefig(os.path.abspath(os.path.join(directory, 'results', name + '_dynamics.png')))
    plt.close()

    plt.stackplot(indexes.keys(), topics_weight_percent)
    plt.xlabel('Year')
    plt.ylabel('Proportion (%)')
    plt.savefig(os.path.abspath(os.path.join(directory, 'results', name + '_dynamics_percent.png')))
    plt.close()


def save_all_data(model: artm.artm_model.ARTM, directory: str, name: str, number_of_topics: int) -> None:
    """
    Save the parameters of the model.
    :param model: the model.
    :param directory: the directory with the dataset.
    :param name: name of the processed dataset.
    :param number_of_topics: the number of topics.
    :return: None.
    """
    if not os.path.exists(os.path.abspath(os.path.join(directory, 'results'))):
        os.makedirs(os.path.abspath(os.path.join(directory, 'results')))
    save_parameters(model, directory, name)
    save_most_popular_tokens(model, directory, name)
    phi_matrix, theta_matrix = save_matrices(model, directory, name)
    save_most_topical_files(number_of_topics, theta_matrix, directory, name)
    save_dynamics(directory, name)
