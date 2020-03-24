"""
Topic modeling related functionality.
"""
import csv
from operator import itemgetter
import os
from typing import Any, Callable, List, Tuple

import artm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .parsing import parse_slice_line, parse_token_line


def check_output_directory(output_dir: str) -> Callable[[Any, str], Any]:
    """
    Check that an argument of the function that represents a directory exists and is a directory.
    :param output_dir: the name of the argument that represents a path to the directory.
    :return: the decorator that checks that the argument
    with the given name exists and is a directory.
    """

    def inner_decorator(fn):
        def wrapper(*args, **kwargs):
            assert os.path.exists(kwargs[output_dir])
            assert os.path.isdir(kwargs[output_dir])
            return fn(*args, **kwargs)

        return wrapper

    return inner_decorator


def create_batches(directory: str, name: str) -> Tuple[artm.BatchVectorizer, artm.Dictionary]:
    """
    Create the batches and the dictionary from the dataset.
    :param directory: the directory with the dataset.
    :param name: name of the processed dataset.
    :return: BatchVectorizer and Dictionary.
    """
    print("Creating the batches and the dictionary of the data.")
    batch_vectorizer = artm.BatchVectorizer(data_path=directory, data_format="bow_uci",
                                            collection_name=name, target_folder=os.path.abspath(
            os.path.join(directory, name + "_batches")))
    dictionary = batch_vectorizer.dictionary
    return batch_vectorizer, dictionary


def define_model(n_topics: int, dictionary: artm.Dictionary, sparse_theta: float,
                 sparse_phi: float,
                 decorrelator_phi: float) -> artm.artm_model.ARTM:
    """
    Define the ARTM model.
    :param n_topics: number of topics.
    :param dictionary: batch vectorizer dictionary.
    :param sparse_theta: sparse theta parameter.
    :param sparse_phi: sparse phi Parameter.
    :param decorrelator_phi: decorellator phi Parameter.
    :return: ARTM model.
    """
    print("Defining the model.")
    topic_names = ["topic_{}".format(i) for i in range(1, n_topics + 1)]
    model_artm = artm.ARTM(topic_names=topic_names, cache_theta=True,
                           scores=[artm.PerplexityScore(name="PerplexityScore",
                                                        dictionary=dictionary),
                                   artm.SparsityPhiScore(name="SparsityPhiScore"),
                                   artm.SparsityThetaScore(name="SparsityThetaScore"),
                                   artm.TopicKernelScore(name="TopicKernelScore",
                                                         probability_mass_threshold=0.3),
                                   artm.TopTokensScore(name="TopTokensScore", num_tokens=15)],
                           regularizers=[artm.SmoothSparseThetaRegularizer(name="SparseTheta",
                                                                           tau=sparse_theta),
                                         artm.SmoothSparsePhiRegularizer(name="SparsePhi",
                                                                         tau=sparse_phi),
                                         artm.DecorrelatorPhiRegularizer(name="DecorrelatorPhi",
                                                                         tau=decorrelator_phi)])
    return model_artm


def train_model(model: artm.artm_model.ARTM, n_doc_iter: int, n_col_iter: int,
                dictionary: artm.Dictionary, batch_vectorizer: artm.BatchVectorizer) -> None:
    """
    Train the ARTM model.
    :param model: the trained model.
    :param n_doc_iter: number of document passes.
    :param n_col_iter: number of collection passes.
    :param dictionary: Batch Vectorizer dictionary.
    :param batch_vectorizer: Batch Vectorizer.
    :return: None.
    """
    print("Training the model.")
    model.num_document_passes = n_doc_iter
    model.initialize(dictionary=dictionary)
    model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=n_col_iter)


@check_output_directory(output_dir="output_dir")
def save_parameters(model: artm.artm_model.ARTM, output_dir: str, name: str) -> None:
    """
    Save the parameters of the model: sparsity phi, sparsity theta, kernel contrast,
    kernel purity, perplexity, and graphs of sparsity phi, sparsity theta, and perplexity.
    When run several times, overwrites the data.
    :param model: the model.
    :param output_dir: the output directory.
    :param name: name of the processed dataset.
    :return: None.
    """
    with open(os.path.abspath(os.path.join(output_dir, name + "_parameters.txt")), "w+") as fout:
        fout.write("Sparsity Phi: {0:.3f}".format(
            model.score_tracker["SparsityPhiScore"].last_value) + "\n")
        fout.write("Sparsity Theta: {0:.3f}".format(
            model.score_tracker["SparsityThetaScore"].last_value) + "\n")
        fout.write("Kernel contrast: {0:.3f}".format(
            model.score_tracker["TopicKernelScore"].last_average_contrast) + "\n")
        fout.write("Kernel purity: {0:.3f}".format(
            model.score_tracker["TopicKernelScore"].last_average_purity) + "\n")
        fout.write("Perplexity: {0:.3f}".format(
            model.score_tracker["PerplexityScore"].last_value) + "\n")

    plt.plot(range(model.num_phi_updates),
             model.score_tracker["PerplexityScore"].value, "r--", linewidth=2)
    plt.xlabel("Iterations count")
    plt.ylabel("Perplexity")
    plt.grid(True)
    plt.savefig(os.path.abspath(os.path.join(output_dir, name + "_perplexity.png")), dpi=1200)
    plt.close()

    plt.plot(range(model.num_phi_updates),
             model.score_tracker["SparsityPhiScore"].value, "r--", linewidth=2)
    plt.xlabel("Iterations count")
    plt.ylabel("Phi Sparsity")
    plt.grid(True)
    plt.savefig(os.path.abspath(os.path.join(output_dir, name + "_phi_sparsity.png")), dpi=1200)
    plt.close()

    plt.plot(range(model.num_phi_updates),
             model.score_tracker["SparsityThetaScore"].value, "r--", linewidth=2)
    plt.xlabel("Iterations count")
    plt.ylabel("Theta Sparsity")
    plt.grid(True)
    plt.savefig(os.path.abspath(os.path.join(output_dir, name + "_theta_sparsity.png")), dpi=1200)
    plt.close()


@check_output_directory(output_dir="output_dir")
def save_most_popular_tokens(model: artm.artm_model.ARTM, output_dir: str, name: str) -> None:
    """
    Save the most popular tokens of the model.
    When run several times, overwrites the data.
    :param model: the model.
    :param output_dir: the output directory.
    :param name: name of the processed dataset.
    :return: None.
    """
    with open(os.path.abspath(os.path.join(output_dir, name + "_most_popular_tokens.txt")),
              "w+") as fout:
        for topic_name in model.topic_names:
            fout.write("{topic_name}: {tokens}\n"
                       .format(topic_name=topic_name,
                               tokens=str(model.score_tracker["TopTokensScore"]
                                          .last_tokens[topic_name])))


@check_output_directory(output_dir="output_dir")
def save_matrices(model: artm.artm_model.ARTM, output_dir: str, name: str) -> None:
    """
    Save the Phi and Theta matrices.
    When run several times, overwrites the data.
    :param model: the model.
    :param output_dir: the output directory.
    :param name: name of the processed dataset.
    :return: Two matrices as DataFrames.
    """
    phi_matrix = model.get_phi().sort_index(axis=0)
    phi_matrix.to_csv(os.path.abspath(os.path.join(output_dir, name + "_phi.csv")))
    theta_matrix = model.get_theta().sort_index(axis=1)
    theta_matrix.to_csv(os.path.abspath(os.path.join(output_dir, name + "_theta.csv")))


@check_output_directory(output_dir="output_dir")
def save_most_topical_files(theta_matrix: pd.DataFrame, tokens_file: str,
                            n_files: int, output_dir: str, name: str) -> None:
    """
    Save the most topical files of the model.
    When run several times, overwrites the data.
    :param theta_matrix: Theta matrix.
    :param tokens_file: the temporary file with tokens.
    :param n_files: number of the most topical files to be saved for each topic.
    :param output_dir: the output directory.
    :param name: name of the processed dataset.
    :return: None.
    """
    file2path = {}
    with open(tokens_file) as fin:
        for line in fin:
            token_line = parse_token_line(line)
            file2path[int(token_line.index)] = token_line.path
    with open(os.path.abspath(os.path.join(output_dir, name + "_most_topical_files.txt")),
              "w+") as fout:
        for i in range(1, theta_matrix.shape[0] + 1):
            fout.write("Topic " + str(i) + "\n\n")
            # Create a dictionary for this topic where keys are files and values are
            # theta values for this topic and this file (n_files largest)
            topic_dict = theta_matrix.sort_values(by="topic_" + str(i), axis=1,
                                                  ascending=False).loc["topic_" +
                                                                       str(i)][:n_files].to_dict()
            for k in topic_dict.keys():
                fout.write("{file_index};{topic_weight:.3f};{file_path}\n"
                           .format(file_index=str(k), topic_weight=topic_dict[k],
                                   file_path=file2path[int(k)]))
            fout.write("\n")


def get_topics_weight(slices_file: str, theta_file: str) -> np.array:
    """
    Read the theta file and transform it into topic weights for different slices.
    :param slices_file: the path to the file with the indices of the slices.
    :param theta_file: the path tp the csv file with the theta matrix.
    :return np.array of weights of each topic for each slice.
    """
    date2indices = {}
    with open(slices_file) as fin:
        for line in fin:
            slice_line = parse_slice_line(line)
            date2indices[slice_line.date] = (slice_line.start_index, slice_line.end_index)
    topics_weight = []
    with open(theta_file) as fin:
        reader = csv.reader(fin)
        next(reader, None)  # Skip the headers
        for row in reader:
            topics_weight.append([])
            for date in date2indices.keys():
                topics_weight[-1].append(
                    sum(float(i) for i in row[date2indices[date][0]:date2indices[date][1] + 1]))
    topics_weight = np.asarray(topics_weight)
    return topics_weight


def get_normalized_dynamics(topics_weight: np.array) -> Tuple[np.array, List]:
    """
    Transform topics weights into normalized topics weights and a list of dynamics parameters
    for every topic: its minimal weight, maximal weight, and their ratio.
    :param topics_weight: numpy array with topic weights.
    :return np.array of normalized weights and a list with dynamics data.
    """
    topics_weight_percent = np.zeros((topics_weight.shape[0], topics_weight.shape[1]))
    for j in range(topics_weight.shape[1]):
        slice_sum = np.sum(topics_weight[:, j], keepdims=True)
        for i in range(topics_weight.shape[0]):
            topics_weight_percent[i, j] = (topics_weight[i, j] / slice_sum) * 100

    dynamics = []
    for i in range(topics_weight_percent.shape[0]):
        dynamics.append(["topic_{}".format(i + 1), min(topics_weight_percent[i]),
                         max(topics_weight_percent[i]),
                         max(topics_weight_percent[i]) / min(topics_weight_percent[i])])
    dynamics = sorted(dynamics, key=itemgetter(3), reverse=True)
    return topics_weight_percent, dynamics


@check_output_directory(output_dir="output_dir")
def save_dynamics(slices_file: str, theta_file: str, output_dir: str, name: str) -> None:
    """
    Save figures with the dynamics.
    When run several times, overwrites the data.
    :param slices_file: the path to the file with the indices of the slices.
    :param theta_file: the path to the csv file with the theta matrix.
    :param output_dir: the output directory.
    :param name: name of the processed dataset.
    :return: None.
    """
    topics_weight = get_topics_weight(slices_file, theta_file)
    topics_weight_percent, dynamics = get_normalized_dynamics(topics_weight)

    np.savetxt(os.path.abspath(os.path.join(output_dir, name + "_dynamics.txt")), topics_weight,
               "%10.3f")
    np.savetxt(os.path.abspath(os.path.join(output_dir, name + "_dynamics_percent.txt")),
               topics_weight_percent, "%10.3f")

    with open(os.path.abspath(os.path.join(output_dir, name + "_dynamics_percent_change.txt")),
              "w+") as fout:
        for topic in dynamics:
            fout.write(
                "{topic_name};{minimum_weight:.3f};{maximum_weight:.3f};{max_min_ratio:.3f}\n"
                .format(topic_name=topic[0], minimum_weight=topic[1],
                        maximum_weight=topic[2], max_min_ratio=topic[3]))

    plt.stackplot(range(1, topics_weight.shape[1] + 1), topics_weight)
    plt.xlabel("Slice")
    plt.ylabel("Proportion (a. u.)")
    plt.savefig(os.path.abspath(os.path.join(output_dir, name + "_dynamics.png")), dpi=1200)
    plt.close()

    for topic in topics_weight.tolist():
        plt.plot(range(1, topics_weight.shape[1] + 1), topic)
    plt.xlabel("Slice")
    plt.ylabel("Proportion (a. u.)")
    plt.savefig(os.path.abspath(os.path.join(output_dir, name + "_dynamics_topics.png")), dpi=1200)
    plt.close()

    plt.stackplot(range(1, topics_weight.shape[1] + 1), topics_weight_percent)
    plt.xlabel("Slice")
    plt.ylabel("Proportion (%)")
    plt.savefig(os.path.abspath(os.path.join(output_dir, name + "_dynamics_percent.png")),
                dpi=1200)
    plt.close()

    for topic in topics_weight_percent.tolist():
        plt.plot(range(1, topics_weight.shape[1] + 1), topic)
    plt.xlabel("Slice")
    plt.ylabel("Proportion (%)")
    plt.savefig(os.path.abspath(os.path.join(output_dir, name + "_dynamics_topics_percent.png")),
                dpi=1200)
    plt.close()


def save_metadata(model: artm.artm_model.ARTM, output_dir: str, name: str, n_files: int) -> None:
    """
    Save the metadata: the parameters of the model, most popular tokens, the matrices,
    most topical files and various dynamics-related statistics.
    :param model: the model.
    :param output_dir: the output directory.
    :param name: name of the processed dataset.
    :param n_files: number of the most topical files to be saved for each topic.
    :return: None.
    """
    print("Saving the results.")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tokens_file = os.path.abspath(os.path.join(output_dir, os.pardir, name + "_tokens.txt"))
    slices_file = os.path.abspath(os.path.join(output_dir, os.pardir, name + "_slices.txt"))
    theta_file = os.path.abspath(os.path.join(output_dir, name + "_theta.csv"))
    theta_matrix = model.get_theta().sort_index(axis=1)

    save_parameters(model=model, output_dir=output_dir, name=name)
    save_most_popular_tokens(model=model, output_dir=output_dir, name=name)
    save_matrices(model=model, output_dir=output_dir, name=name)
    save_most_topical_files(theta_matrix=theta_matrix, tokens_file=tokens_file,
                            n_files=n_files, output_dir=output_dir, name=name)
    save_dynamics(slices_file=slices_file, theta_file=theta_file, output_dir=output_dir, name=name)


def model_topics(output_dir: str, name: str, n_topics: int, sparse_theta: float, sparse_phi: float,
                 decorrelator_phi: float, n_doc_iter: int, n_col_iter: int, n_files: int) -> None:
    """
    Take the input, create the batches, train the model with the given parameters,
    and saves all metadata.
    :param output_dir: the output directory.
    :param name: name of the processed dataset.
    :param n_topics: number of topics.
    :param sparse_theta: sparse theta parameter.
    :param sparse_phi: sparse phi parameter.
    :param decorrelator_phi: decorellator phi parameter.
    :param n_doc_iter: number of document passes.
    :param n_col_iter: number of collection passes.
    :param n_files: number of the most topical files to be saved for each topic.
    :return: None.
    """
    batch_vectorizer, dictionary = create_batches(output_dir, name)
    model = define_model(n_topics, dictionary, sparse_theta, sparse_phi, decorrelator_phi)
    train_model(model, n_doc_iter, n_col_iter, dictionary, batch_vectorizer)
    results_dir = os.path.abspath(os.path.join(output_dir, "results"))
    save_metadata(model, results_dir, name, n_files)
    print("Topic modeling finished.")
