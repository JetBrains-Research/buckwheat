# Copyright 2018 source{d}
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from enum import Enum
import functools
import re

import Stemmer


class TokenStyle(Enum):
    """Metadata that should allow to reconstruct initial identifier from a list of tokens."""
    DELIMITER = 1
    TOKEN_UPPER = 2
    TOKEN_LOWER = 3
    TOKEN_CAPITALIZED = 4


class TokenParser:
    """
    Common utilities for splitting and stemming tokens.
    """
    # Regexp to split source code identifiers
    NAME_BREAKUP_RE = re.compile(r"[^a-zA-Z]+")
    NAME_BREAKUP_KEEP_DELIMITERS_RE = re.compile(r"([^a-zA-Z]+)")  # ... and keep delimiters
    # Example:
    # token = "Var_WithStrangeNAMING__very_strange"
    # NAME_BREAKUP_KEEP_DELIMITERS_RE.split(token) -> ['Var', '_', 'WithStrangeNAMING', '__',
    #                                                  'very', '_', 'strange']
    # NAME_BREAKUP_RE.split(token) -> ['Var', 'WithStrangeNAMING', 'very', 'strange']
    STEM_THRESHOLD = 6  # We do not stem split parts shorter than or equal to this size.
    MAX_TOKEN_LENGTH = 256  # We cut identifiers longer than this value.
    MIN_SPLIT_LENGTH = 3  # We do not split source code identifiers shorter than this value.
    DEFAULT_SINGLE_SHOT = False  # True if we do not want to join small identifiers to next one.
    # Example: 'sourced.ml.algorithms' -> ["sourc", "sourcedml", "algorithm", "mlalgorithm"].
    # if True we have only ["sourc", "algorithm"].
    # if you do not want to filter small tokens set min_split_length=1.
    SAVE_TOKEN_STYLE = False  # whether yield metadata that can be used to reconstruct initial
    # identifier.
    ATTACH_UPPER = True  # True to attach the last of several uppercase letters in a row to

    # the next token. Example: 'HTMLResponse' -> ["html", "response"] if True,
    # 'HTMLResponse' -> ["htmlr", "esponse"] if False.

    def __init__(self, stem_threshold=STEM_THRESHOLD, max_token_length=MAX_TOKEN_LENGTH,
                 min_split_length=MIN_SPLIT_LENGTH, single_shot=DEFAULT_SINGLE_SHOT,
                 save_token_style=SAVE_TOKEN_STYLE, attach_upper=ATTACH_UPPER):
        self._stemmer = Stemmer.Stemmer("english")
        self._stemmer.maxCacheSize = 0
        self._stem_threshold = stem_threshold
        self._max_token_length = max_token_length
        self._min_split_length = min_split_length
        self._single_shot = single_shot
        self._save_token_style = save_token_style
        self._attach_upper = attach_upper
        if self._save_token_style and not self._single_shot:
            raise ValueError("Only one of `single_shot`/`save_token_style` should be True")

    @property
    def stem_threshold(self):
        return self._stem_threshold

    @stem_threshold.setter
    def stem_threshold(self, value):
        if not isinstance(value, int):
            raise TypeError("stem_threshold must be an integer - got %s" % type(value))
        if value < 1:
            raise ValueError("stem_threshold must be greater than 0 - got %d" % value)
        self._stem_threshold = value

    @property
    def max_token_length(self):
        return self._max_token_length

    @max_token_length.setter
    def max_token_length(self, value):
        if not isinstance(value, int):
            raise TypeError("max_token_length must be an integer - got %s" % type(value))
        if value < 1:
            raise ValueError("max_token_length must be greater than 0 - got %d" % value)
        self._max_token_length = value

    @property
    def min_split_length(self):
        return self._min_split_length

    @min_split_length.setter
    def min_split_length(self, value):
        if not isinstance(value, int):
            raise TypeError("min_split_length must be an integer - got %s" % type(value))
        if value < 1:
            raise ValueError("min_split_length must be greater than 0 - got %d" % value)
        self._min_split_length = value

    def __call__(self, token):
        return self.process_token(token)

    def process_token(self, token):
        for word in self.split(token):
            yield self.stem(word)

    def stem(self, word):
        if len(word) <= self.stem_threshold:
            return word
        return self._stemmer.stemWord(word)

    def split(self, token):
        token = token.strip()[:self.max_token_length]

        def meta_decorator(func):
            if self._save_token_style:
                @functools.wraps(func)
                def decorated_func(name):
                    if name.isupper():
                        meta = TokenStyle.TOKEN_UPPER
                    elif name.islower():
                        meta = TokenStyle.TOKEN_LOWER
                    else:
                        meta = TokenStyle.TOKEN_CAPITALIZED
                    for res in func(name):
                        yield res, meta

                return decorated_func
            else:
                return func

        @meta_decorator
        def ret(name):
            r = name.lower()
            if len(name) >= self.min_split_length:
                yield r
                if ret.prev_p and not self._single_shot:
                    yield ret.prev_p + r
                    ret.prev_p = ""
            elif not self._single_shot:
                ret.prev_p = r

        ret.prev_p = ""

        if self._save_token_style:
            regexp_splitter = self.NAME_BREAKUP_KEEP_DELIMITERS_RE
        else:
            regexp_splitter = self.NAME_BREAKUP_RE

        for part in regexp_splitter.split(token):
            if not part:
                continue
            if self._save_token_style and not part.isalpha():
                yield part, TokenStyle.DELIMITER
                continue
            assert part.isalpha()
            start = 0
            for i in range(1, len(part)):
                this = part[i]
                prev = part[i - 1]
                if prev.islower() and this.isupper():
                    yield from ret(part[start:i])
                    start = i
                elif prev.isupper() and this.islower():
                    if self._attach_upper and i > 1 and part[i - 2].isupper():
                        new_start = i - 1
                    else:
                        new_start = i
                    if i - 1 > start:
                        yield from ret(part[start:new_start])
                        start = new_start
            last = part[start:]
            if last:
                yield from ret(last)

    @staticmethod
    def reconstruct(tokens):
        res = []
        for t, meta in tokens:
            if meta == TokenStyle.DELIMITER:
                res.append(t)
            elif meta == TokenStyle.TOKEN_LOWER:
                res.append(t.lower())
            elif meta == TokenStyle.TOKEN_UPPER:
                res.append(t.upper())
            elif meta == TokenStyle.TOKEN_CAPITALIZED:
                res.append(t[0].upper() + t[1:])
        return "".join(res)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_stemmer"]
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._stemmer = Stemmer.Stemmer("english")
