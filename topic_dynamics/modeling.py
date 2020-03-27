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
from scipy.stats import entropy

from .parsing import parse_slice_line, parse_token_line


def check_output_directory(output_dir: str) -> Callable[[Any, str], Any]:
    """
    Check that an argument of the function that represents a directory exists and is a directory.
    :param output_dir: the name of the argument that represents a path to the directory.
    :return: the decorator that checks that the argument with the given name exists and is a
             directory.
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
    batch_vectorizer = artm.BatchVectorizer(
        data_path=directory, data_format="bow_uci", collection_name=name,
        target_folder=os.path.abspath(os.path.join(directory, "batches")))
    dictionary = batch_vectorizer.dictionary
    return batch_vectorizer, dictionary


def define_model(n_topics: int, dictionary: artm.Dictionary, sparse_theta: float,
                 sparse_phi: float, decorrelator_phi: float) -> artm.artm_model.ARTM:
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
def save_parameters(model: artm.artm_model.ARTM, output_dir: str) -> None:
    """
    Save the parameters of the model: sparsity phi, sparsity theta, kernel contrast,
    kernel purity, perplexity, and graphs of sparsity phi, sparsity theta, and perplexity.
    When run several times, overwrites the data.
    :param model: the model.
    :param output_dir: the output directory.
    :return: None.
    """
    with open(os.path.abspath(os.path.join(output_dir, "metrics.txt")), "w+") as fout:
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
             model.score_tracker["PerplexityScore"].value, linewidth=2)
    plt.xlabel("Iterations count")
    plt.ylabel("Perplexity")
    plt.grid(True)
    plt.savefig(os.path.abspath(os.path.join(output_dir, "perplexity.png")), dpi=600,
                bbox_inches="tight")
    plt.close()

    plt.plot(range(model.num_phi_updates),
             model.score_tracker["SparsityPhiScore"].value, linewidth=2)
    plt.xlabel("Iterations count")
    plt.ylabel("Phi Sparsity")
    plt.grid(True)
    plt.savefig(os.path.abspath(os.path.join(output_dir, "phi_sparsity.png")), dpi=600,
                bbox_inches="tight")
    plt.close()

    plt.plot(range(model.num_phi_updates),
             model.score_tracker["SparsityThetaScore"].value, linewidth=2)
    plt.xlabel("Iterations count")
    plt.ylabel("Theta Sparsity")
    plt.grid(True)
    plt.savefig(os.path.abspath(os.path.join(output_dir, "theta_sparsity.png")), dpi=600,
                bbox_inches="tight")
    plt.close()


@check_output_directory(output_dir="output_dir")
def save_most_popular_tokens(model: artm.artm_model.ARTM, output_dir: str) -> None:
    """
    Save the most popular tokens of the model.
    When run several times, overwrites the data.
    :param model: the model.
    :param output_dir: the output directory.
    :return: None.
    """
    with open(os.path.abspath(os.path.join(output_dir, "most_popular_tokens.txt")),
              "w+") as fout:
        for topic_name in model.topic_names:
            fout.write("{topic_name}: {tokens}\n"
                       .format(topic_name=topic_name,
                               tokens=str(model.score_tracker["TopTokensScore"]
                                          .last_tokens[topic_name])))


@check_output_directory(output_dir="output_dir")
def save_matrices(model: artm.artm_model.ARTM, output_dir: str) -> None:
    """
    Save the Phi and Theta matrices.
    When run several times, overwrites the data.
    :param model: the model.
    :param output_dir: the output directory.
    :return: Two matrices as DataFrames.
    """
    phi_matrix = model.get_phi().sort_index(axis=0)
    phi_matrix.to_csv(os.path.abspath(os.path.join(output_dir, "phi.csv")))
    theta_matrix = model.get_theta().sort_index(axis=1)
    theta_matrix.to_csv(os.path.abspath(os.path.join(output_dir, "theta.csv")))


@check_output_directory(output_dir="output_dir")
def save_most_topical_files(theta_matrix: pd.DataFrame, tokens_file: str,
                            n_files: int, output_dir: str) -> None:
    """
    Save the most topical files of the model.
    When run several times, overwrites the data.
    :param theta_matrix: Theta matrix.
    :param tokens_file: the temporary file with tokens.
    :param n_files: number of the most topical files to be saved for each topic.
    :param output_dir: the output directory.
    :return: None.
    """
    file2path = {}
    with open(tokens_file) as fin:
        for line in fin:
            token_line = parse_token_line(line)
            file2path[int(token_line.index)] = token_line.path
    with open(os.path.abspath(os.path.join(output_dir, "most_topical_files.txt")),
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


def get_topics_metrics(slices_file: str, theta_file: str) -> Tuple[np.array, np.array,
                                                                   np.array, np.array]:
    """
    Read the theta file and transform it into topic metrics for different slices:
    assignments and focuses.
    :param slices_file: the path to the file with the indices of the slices.
    :param theta_file: the path tp the csv file with the theta matrix.
    :return np.arrays of assignments and focuses of each topic for each slice.
    """
    date2indices = {}
    with open(slices_file) as fin:
        for line in fin:
            slice_line = parse_slice_line(line)
            date2indices[slice_line.date] = (slice_line.start_index, slice_line.end_index)
    assignment = []
    assignment_normalized = []
    scattering = []
    focus = []
    with open(theta_file) as fin:
        reader = csv.reader(fin)
        next(reader, None)  # Skip the headers
        for row in reader:
            assignment.append([])
            assignment_normalized.append([])
            scattering.append([])
            focus.append([])
            for date in date2indices.keys():
                version_row = [float(i) for i in row[date2indices[date][0]:
                                                     date2indices[date][1] + 1]]
                assignment[-1].append(sum(version_row))
                assignment_normalized[-1].append(100 * assignment[-1][-1] / len(version_row))
                scattering[-1].append(entropy(version_row))
                focus[-1].append(100 * sum(1 for i in version_row if i >= 0.5) / len(version_row))
    assignment = np.asarray(assignment)
    assignment_normalized = np.asarray(assignment_normalized)
    scattering = np.asarray(scattering)
    focus = np.asarray(focus)
    return assignment, assignment_normalized, scattering, focus


@check_output_directory(output_dir="output_dir")
def save_metric(array: np.array, name: str, x_label: str, y_label: str, output_dir: str) -> None:
    """
    Saves 2D np.array of a metric (topics vs slices) as a text file and as a stacked plot with a
    given name and axes labels into a given output directory.
    :param array: 2D np.array with the necessary information.
    :param name: name of the files to be saved.
    :param x_label: x axis label for the stacked plot.
    :param y_label: y axis label for the stacked plot.
    :param output_dir: the path to the output directory.
    :return: None.
    """
    np.savetxt(os.path.abspath(os.path.join(output_dir, name + ".txt")), array, "%10.3f")
    plt.stackplot(range(1, array.shape[1] + 1), array)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(os.path.abspath(os.path.join(output_dir, name + ".png")), dpi=600,
                bbox_inches="tight")
    plt.close()


@check_output_directory(output_dir="output_dir")
def save_metric_change(metric: np.array, output_dir: str, output_file: str) -> None:
    """
    Save lists of parameters for the metric of every topic:
    its minimal value, maximal value, and their ratio.
    :param metric: numpy array with a given metric.
    :param output_dir: the path to the output directory.
    :param output_file: the name of the output file.
    :return None.
    """
    dynamics = []
    for i in range(metric.shape[0]):
        dynamics.append(["topic_{}".format(i + 1), min(metric[i]), max(metric[i]),
                         max(metric[i]) / min(metric[i])])
    dynamics = sorted(dynamics, key=itemgetter(3), reverse=True)
    with open(os.path.abspath(os.path.join(output_dir, output_file)), "w+") as fout:
        for topic in dynamics:
            fout.write(
                "{topic_name};{min_assignment:.3f};{max_assignment:.3f};{max_min_ratio:.3f}\n"
                    .format(topic_name=topic[0], min_assignment=topic[1],
                            max_assignment=topic[2], max_min_ratio=topic[3]))


@check_output_directory("output_dir")
def save_topics_change(assignment: np.array, assignment_normalized: np.array,
                       scattering: np.array, focus: np.array, output_dir: str) -> None:
    """
    Create a combined plot of various metrics for every topic and save them to the given directory.
    :param assignment: numpy array with assignment values in different slices for a given topic.
    :param assignment_normalized: numpy array with normalized assignment values in different slices
           for a given topic.
    :param scattering: numpy array with scattering values in different slices for a given topic.
    :param focus: numpy array with focus values in different slices for a given topic.
    :param output_dir: the path to the output directory.
    :return None.
    """
    for topic_number in range(assignment.shape[0]):
        fig, axs = plt.subplots(4, figsize=([5, 10]))
        axs[0].set_title("Topic {topic_number}".format(topic_number=topic_number + 1))
        axs[0].plot(range(1, assignment.shape[1] + 1), assignment[topic_number], "tab:red")
        axs[0].set_ylabel("Assignment (a. u.)")
        axs[0].grid(b=True)
        axs[1].plot(range(1, assignment_normalized.shape[1] + 1),
                    assignment_normalized[topic_number], "tab:green")
        axs[1].set_ylabel("Normalized assignment (%)")
        axs[1].grid(b=True)
        axs[2].plot(range(1, scattering.shape[1] + 1), scattering[topic_number], "tab:blue")
        axs[2].set_ylabel("Scattering (a. u.)")
        axs[2].grid(b=True)
        axs[3].plot(range(1, focus.shape[1] + 1), focus[topic_number], "tab:purple")
        axs[3].set_ylabel("Focus (%)")
        axs[3].grid(b=True)
        axs[3].set_xlabel("Slice")
        fig.align_ylabels(axs)
        plt.subplots_adjust(hspace=0.5)
        plt.savefig(os.path.abspath(os.path.join(output_dir,
                                                 "topic_{topic_number}.png"
                                                 .format(topic_number=topic_number + 1))),
                    dpi=600, bbox_inches="tight")
        plt.close()


def save_dynamics(slices_file: str, theta_file: str, output_dir: str) -> None:
    """
    Save figures with the dynamics.
    When run several times, overwrites the data.
    :param slices_file: the path to the file with the indices of the slices.
    :param theta_file: the path to the csv file with the theta matrix.
    :param output_dir: the output directory.
    :return: None.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    assignment, assignment_normalized, scattering, focus = get_topics_metrics(slices_file,
                                                                              theta_file)
    save_metric(array=assignment, name="assignment", x_label="Slice",
                y_label="Assignment (a. u.)", output_dir=output_dir)
    save_metric(array=assignment_normalized, name="assignment_normalized", x_label="Slice",
                y_label="Normalized assignment (%)", output_dir=output_dir)
    save_metric(array=scattering, name="scattering", x_label="Slice", y_label="Scatter (a. u.)",
                output_dir=output_dir)
    save_metric(array=focus, name="focus", x_label="Slice", y_label="Focus (%)",
                output_dir=output_dir)
    save_metric_change(metric=assignment, output_dir=output_dir,
                       output_file="assignment_change.txt")
    save_metric_change(metric=assignment_normalized, output_dir=output_dir,
                       output_file="assignment_normalized_change.txt")
    save_metric_change(metric=scattering, output_dir=output_dir,
                       output_file="scattering_change.txt")
    save_metric_change(metric=focus, output_dir=output_dir, output_file="focus_change.txt")
    save_topics_change(assignment=assignment, assignment_normalized=assignment_normalized,
                       scattering=scattering, focus=focus, output_dir=output_dir)


def save_metadata(model: artm.artm_model.ARTM, output_dir: str, n_files: int,
                  tokens_file: str, slices_file: str) -> None:
    """
    Save the metadata: the parameters of the model, most popular tokens, the matrices,
    most topical files and various dynamics-related statistics.
    :param model: the model.
    :param output_dir: the output directory.
    :param n_files: number of the most topical files to be saved for each topic.
    :param tokens_file: the temporary file with tokens.
    :param slices_file: the path to the file with the indices of the slices.
    :return: None.
    """
    print("Saving the results.")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    theta_file = os.path.abspath(os.path.join(output_dir, "theta.csv"))
    theta_matrix = model.get_theta().sort_index(axis=1)
    dynamics_dir = os.path.abspath(os.path.join(output_dir, "dynamics"))

    save_parameters(model=model, output_dir=output_dir)
    save_most_popular_tokens(model=model, output_dir=output_dir)
    save_matrices(model=model, output_dir=output_dir)
    save_most_topical_files(theta_matrix=theta_matrix, tokens_file=tokens_file,
                            n_files=n_files, output_dir=output_dir)
    save_dynamics(slices_file=slices_file, theta_file=theta_file, output_dir=dynamics_dir)


def model_topics(output_dir: str, n_topics: int, sparse_theta: float, sparse_phi: float,
                 decorrelator_phi: float, n_doc_iter: int, n_col_iter: int, n_files: int,
                 diffs: bool) -> None:
    """
    Take the input, create the batches, train the model with the given parameters,
    and saves all metadata.
    :param output_dir: the output directory.
    :param n_topics: number of topics.
    :param sparse_theta: sparse theta parameter.
    :param sparse_phi: sparse phi parameter.
    :param decorrelator_phi: decorellator phi parameter.
    :param n_doc_iter: number of document passes.
    :param n_col_iter: number of collection passes.
    :param n_files: number of the most topical files to be saved for each topic.
    :param diffs: True if the topics are modeled on diffs,
                  False if they are modeled on full files.
    :return: None.
    """
    if diffs:
        name = "diffs_dataset"
        tokens_file = os.path.abspath(os.path.join(output_dir, "diffs_tokens.txt"))
        slices_file = os.path.abspath(os.path.join(output_dir, "diffs_slices.txt"))
    else:
        name = "dataset"
        tokens_file = os.path.abspath(os.path.join(output_dir, "tokens.txt"))
        slices_file = os.path.abspath(os.path.join(output_dir, "slices.txt"))
    batch_vectorizer, dictionary = create_batches(output_dir, name)
    model = define_model(n_topics, dictionary, sparse_theta, sparse_phi, decorrelator_phi)
    train_model(model, n_doc_iter, n_col_iter, dictionary, batch_vectorizer)
    results_dir = os.path.abspath(os.path.join(output_dir, "results"))
    save_metadata(model, results_dir, n_files, tokens_file, slices_file)
    print("Topic modeling finished.")
