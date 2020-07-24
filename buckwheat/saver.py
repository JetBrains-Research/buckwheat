"""
Output-related functionality
"""
from collections import Counter
import dataclasses
import json
from operator import itemgetter
import os
from typing import Dict, List, Union

from .utils import FileData, IdentifierData, IdentifiersTypes, ObjectTypes


def merge_bags(files: List[FileData]) -> Counter:
    """
    Transform sequences of identifiers in FileData objects into Counter objects and merge them.
    :param files: a list of FileData objects.
    :return: a Counter object of tokens and their counts..
    """
    repository_tokens = Counter()
    for file in files:
        repository_tokens += Counter(file.identifiers)
    return repository_tokens


class OutputFormats:
    def __init__(self, output_format: str,
                 reps2files: Dict[str, List[FileData]],
                 mode: str, gran: str, output_dir: str, filename: str):
        if output_format == "wabbit":
            self.save_wabbit(reps2files, mode, gran, output_dir, filename)
        elif output_format == "json":
            self.save_json(reps2files, mode, gran, output_dir, filename)

    @classmethod
    def save_wabbit(cls, reps2files: Dict[str, List[FileData]], mode: str, gran: str,
                    output_dir: str, filename: str) -> None:
        """
        Save the bags of tokens in the Vowpal Wabbit format: one bag per line, in the format
        "name token1:parameters token2:parameters...". When run again, overwrites the data.
        :param reps2files: a dictionary with repositories names as keys and the lists of
                           FileData objects as values.
        :param mode: the mode of parsing. Either "counters" or "sequences".
        :param gran: granularity of parsing. Values are ["projects", "files", "classes",
                     "functions"].
        :param output_dir: full path to the output directory.
        :param filename: the name of the output file.
        :return: none.
        """

        def counter_to_wabbit(tokens_counter: Counter) -> str:
            """
            Transforms a Counter object into a saving format of Wabbit:
            "token1:count1, token2:count2..."
            :param tokens_counter: a Counter object of tokens and their count.
            :return: string "token1:count1, token2:count2..." sorted by descending count.
            """
            sorted_tokens = sorted(tokens_counter.items(), key=itemgetter(1), reverse=True)
            formatted_tokens = []
            for token in sorted_tokens:
                formatted_tokens.append("{token}:{count}"
                                        .format(token=token[0], count=str(token[1])))
            return " ".join(formatted_tokens)

        def sequence_to_wabbit(sequence: Union[List[str], List[IdentifierData]],
                               identifiers_type: IdentifiersTypes) -> str:
            """
            Transforms a sequence of tokens and their parameters into a saving format of Wabbit:
            "token1:parameters token2:parameters...".
            :param sequence: a list of tokens as either strings or IdentifierData objects.
            :param identifiers_type: type of the sequence.
            :return: string "token1:parameters token2:parameters..." sorted as in original code.
            """
            if identifiers_type == IdentifiersTypes.STRING:
                return " ".join(sequence)
            elif identifiers_type == IdentifiersTypes.VERBOSE:
                formatted_tokens = []
                for token in sequence:
                    parameters = ",".join([str(parameter) for parameter in
                                           [token.start_byte, token.start_line,
                                            token.start_column]])
                    formatted_tokens.append("{token}:{parameters}".format(token=token.identifier,
                                                                          parameters=parameters))
                return " ".join(formatted_tokens)

        with open(os.path.abspath(os.path.join(output_dir, filename)), "w+") as fout:
            if gran == "projects":
                # If the granularity is "projects", all identifiers for each project are merged.
                for repository_name in reps2files.keys():
                    repository_tokens = counter_to_wabbit(merge_bags(reps2files[repository_name]))
                    if len(repository_tokens) != 0:  # Skipping empty repositories.
                        fout.write("{name} {tokens}\n"
                                   .format(name=repository_name,
                                           tokens=repository_tokens))
            elif gran == "files":
                for repository_name in reps2files.values():
                    for file in repository_name:
                        if mode == "counters":
                            file_tokens = counter_to_wabbit(Counter(file.identifiers))
                        else:
                            file_tokens = sequence_to_wabbit(file.identifiers,
                                                             file.identifiers_type)
                        if len(file_tokens) != 0:  # Skipping empty files.
                            fout.write("{name} {tokens}\n"
                                       .format(name=file.path,
                                               tokens=file_tokens))
            else:
                for repository_name in reps2files.values():
                    for file in repository_name:
                        for obj in file.objects:
                            if (gran == "functions" and
                                obj.object_type == ObjectTypes.FUNCTION) or \
                                    (gran == "classes" and obj.object_type == ObjectTypes.CLASS):
                                if mode == "counters":
                                    object_tokens = counter_to_wabbit(Counter(obj.identifiers))
                                else:
                                    object_tokens = sequence_to_wabbit(obj.identifiers,
                                                                       obj.identifiers_type)
                                if len(object_tokens) != 0:  # Skipping empty objects.
                                    fout.write("{name} {tokens}\n"
                                               .format(name=f"{file.path}"
                                                            f"#L{obj.start_line + 1}-"
                                                            f"L{obj.end_line + 1}",
                                                       tokens=object_tokens))

    @classmethod
    def save_json(cls, reps2files: Dict[str, List[FileData]], mode: str, gran: str,
                  output_dir: str, filename: str) -> None:
        """
        Save the bags of tokens as JSON files
        :param reps2files: a dictionary with repositories names as keys and their bags of tokens as
                           values. The bags are also dictionaries with bags' names as keys and
                           sequences of tokens and their parameters as values.
        :param mode: the mode of parsing. Either "counters" or "sequences".
        :param gran: granularity of parsing. Values are ["projects", "files", "classes",
                     "functions"].
        :param output_dir: full path to the output directory.
        :param filename: the name of the output file.
        :return: none.
        """
        res = {}
        with open(os.path.abspath(os.path.join(output_dir, filename)), "w+") as fout:
            if gran == "projects":
                for repository_name in reps2files.keys():
                    repository_tokens = merge_bags(reps2files[repository_name])
                    if len(repository_tokens) != 0:  # Skipping empty repositories.
                        res[repository_name] = repository_tokens
            elif gran == "files":
                for repository_name in reps2files.keys():
                    res[repository_name] = {}
                    for file in reps2files[repository_name]:
                        if len(file.identifiers) != 0:  # Skipping empty files.
                            if mode == "counters":
                                res[repository_name][file.path] = Counter(file.identifiers)
                            else:
                                if file.identifiers_type == IdentifiersTypes.STRING:
                                    res[repository_name][file.path] = file.identifiers
                                elif file.identifiers_type == IdentifiersTypes.VERBOSE:
                                    tokens = []
                                    for identifier in file.identifiers:
                                        tokens.append(dataclasses.astuple(identifier))
                                    res[repository_name][file.path] = tokens
            else:
                for repository_name in reps2files.keys():
                    res[repository_name] = {}
                    for file in reps2files[repository_name]:
                        for obj in file.objects:
                            if len(obj.identifiers) != 0:  # Skipping empty objects.
                                if (gran == "functions" and obj.object_type == ObjectTypes
                                        .FUNCTION) or (gran == "classes" and
                                                       obj.object_type == ObjectTypes.CLASS):
                                    if mode == "counters":
                                        res[repository_name][
                                            f"{file.path}#L{obj.start_line + 1}-"
                                            f"L{obj.end_line + 1}"] = Counter(obj.identifiers)
                                    else:
                                        if obj.identifiers_type == IdentifiersTypes.STRING:
                                            res[repository_name][
                                                f"{file.path}#L{obj.start_line + 1}-"
                                                f"L{obj.end_line + 1}"] = obj.identifiers
                                        elif obj.identifiers_type == IdentifiersTypes.VERBOSE:
                                            tokens = []
                                            for identifier in obj.identifiers:
                                                tokens.append(dataclasses.astuple(identifier))
                                            res[repository_name][
                                                f"{file.path}#L{obj.start_line + 1}-"
                                                f"L{obj.end_line + 1}"] = tokens
            json.dump(res, fout, ensure_ascii=False, indent=4)
