"""
Output-related functionality
"""
from collections import Counter
import json
from operator import itemgetter
import os
from typing import Dict, List, Tuple


def sequence_to_counter(sequence: List[Tuple[str, int, int, int]]) -> Counter:
    """
    Transforms a list of tuples with tokens and their information into a counter object.
    :param sequence: a list of tuples, where the first element of the tuple is a token.
    :return: a Counter object of the tokens and their counts.
    """
    tokens = []
    for token in sequence:
        tokens.append(token[0])
    return Counter(tokens)


def merge_bags(bag2tokens: Dict[str, List[Tuple[str, int, int, int]]]) -> Counter:
    """
    Transform the sequences of tokens into counters and merge them for projects.
    :param bag2tokens: a dictionary with bags' names as keys and sequences of tokens and their
                       parameters as values.
    :return: a Counter object of tokens and their counts..
    """
    repository_tokens = Counter()
    for bag_tokens in bag2tokens.values():
        repository_tokens += sequence_to_counter(bag_tokens)
    return repository_tokens


class OutputFormats:
    def __init__(self, output_format: str,
                 reps2bags: Dict[str, Dict[str, List[Tuple[str, int, int, int]]]],
                 mode: str, gran: str, output_dir: str, filename: str):
        if output_format == "wabbit":
            self.save_wabbit(reps2bags, mode, gran, output_dir, filename)
        elif output_format == "json":
            self.save_json(reps2bags, mode, gran, output_dir, filename)

    @classmethod
    def save_wabbit(cls, reps2bags: Dict[str, Dict[str, List[Tuple[str, int, int, int]]]],
                    mode: str, gran: str, output_dir: str, filename: str) -> None:
        """
        Save the bags of tokens in the Vowpal Wabbit format: one bag per line, in the format
        "name token1:parameters token2:parameters...". When run again, overwrites the data.
        :param reps2bags: a dictionary with repositories names as keys and their bags of tokens as
                          values. The bags are also dictionaries with bags' names as keys and
                          sequences of tokens and their parameters as values.
        :param mode: The mode of parsing. 'counters' returns Counter objects of subtokens and their
                     count, 'sequences' returns full sequences of subtokens and their parameters:
                     starting byte, starting line, starting symbol in line, ending symbol in line.
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
            :return: string "token1:count1, token2:count2..." sorted alphabetically.
            """
            sorted_tokens = sorted(tokens_counter.items(), key=itemgetter(1))
            formatted_tokens = []
            for token in sorted_tokens:
                formatted_tokens.append("{token}:{count}"
                                        .format(token=token[0], count=str(token[1])))
            return " ".join(formatted_tokens)

        def sequence_to_wabbit(sequence: List[Tuple[str, int, int, int]]) -> str:
            """
            Transforms a sequence of tokens and their parameters into a saving format of Wabbit:
            "token1:parameters token2:parameters...".
            :param sequence: a list of tokens and their parameters - starting byte, starting line,
            starting symbol in line.
            :return: string "token1:parameters token2:parameters..." sorted as in original code.
            """
            formatted_tokens = []
            for token in sequence:
                parameters = ",".join([str(parameter) for parameter in token[1:]])
                formatted_tokens.append("{token}:{parameters}".format(token=token[0],
                                                                      parameters=parameters))
            return " ".join(formatted_tokens)

        with open(os.path.abspath(os.path.join(output_dir, filename)), "w+") as fout:
            # If the granularity is 'projects', all the bags for each project are merged into one.
            # Only Counter mode available (sequence is meaningless for the entire project).
            if gran == "projects":
                for repository_name in reps2bags.keys():
                    repository_tokens = merge_bags(reps2bags[repository_name])
                    fout.write("{name} {tokens}\n"
                               .format(name=repository_name,
                                       tokens=counter_to_wabbit(repository_tokens)))
            # If the granularity is 'files' or finer, then each bag is saved individually.
            else:
                for repository_name in reps2bags.keys():
                    for bag_name in reps2bags[repository_name].keys():
                        if mode == "counters":
                            tokens = counter_to_wabbit(sequence_to_counter(
                                               reps2bags[repository_name][bag_name]))
                        elif mode == "sequences":
                            tokens = sequence_to_wabbit(reps2bags[repository_name][bag_name])
                        fout.write("{name} {tokens}\n"
                                   .format(name=bag_name, tokens=tokens))

    @classmethod
    def save_json(cls, reps2bags: Dict[str, Dict[str, List[Tuple[str, int, int, int]]]],
                  mode: str, gran: str, output_dir: str, filename: str) -> None:
        """
        Save the bags of tokens as JSON files
        :param reps2bags: a dictionary with repositories names as keys and their bags of tokens as
                          values. The bags are also dictionaries with bags' names as keys and
                          sequences of tokens and their parameters as values.
        :param mode: The mode of parsing. 'counters' returns Counter objects of subtokens and their
                     count, 'sequences' returns full sequences of subtokens and their parameters:
                     starting byte, starting line, starting symbol in line, ending symbol in line.
        :param gran: granularity of parsing. Values are ["projects", "files", "classes",
                     "functions"].
        :param output_dir: full path to the output directory.
        :param filename: the name of the output file.
        :return: none.
        """
        with open(os.path.abspath(os.path.join(output_dir, filename)), "w+") as fout:
            if gran == "projects":
                for repository_name in reps2bags.keys():
                    reps2bags[repository_name] = merge_bags(reps2bags[repository_name])
                json.dump(reps2bags, fout, ensure_ascii=False, indent=4)
            else:
                if mode == "counters":
                    for repository_name in reps2bags.keys():
                        for bag_name in reps2bags[repository_name].keys():
                            reps2bags[repository_name][bag_name] = sequence_to_counter(
                                reps2bags[repository_name][bag_name])
                json.dump(reps2bags, fout, ensure_ascii=False, indent=4)
