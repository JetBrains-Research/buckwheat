"""
Tree-sitter related functionality.
"""
import os

from tree_sitter import Language, Parser

PARSERS = {}


def get_tree_sitter_dir() -> str:
    """
    Get tree-sitter directory.
    :return: absolute path.
    """
    return os.path.abspath(os.path.dirname(__file__))


def get_tree_sitter_so() -> str:
    """
    Get build tree-sitter `.so` location.
    :return: absolute path.
    """
    tree_sitter_dir = get_tree_sitter_dir()
    bin_loc = os.path.join(tree_sitter_dir, "build", "langs.so")
    return bin_loc


def main() -> None:
    """
    Initialize tree-sitter library.
    :return: None.
    """
    # root directory for tree-sitter
    tree_sitter_dir = get_tree_sitter_dir()
    # grammar locations
    javascript_grammar_loc = os.path.join(tree_sitter_dir, "vendor", "tree-sitter-javascript")
    java_grammar_loc = os.path.join(tree_sitter_dir, "vendor", "tree-sitter-java")
    python_grammar_loc = os.path.join(tree_sitter_dir, "vendor", "tree-sitter-python")
    go_grammar_loc = os.path.join(tree_sitter_dir, "vendor", "tree-sitter-go")
    cpp_grammar_loc = os.path.join(tree_sitter_dir, "vendor", "tree-sitter-cpp")
    ruby_grammar_loc = os.path.join(tree_sitter_dir, "vendor", "tree-sitter-ruby")
    typescript_grammar_loc = os.path.join(tree_sitter_dir,
                                          "vendor", "tree-sitter-typescript", "typescript")
    tsx_grammar_loc = os.path.join(tree_sitter_dir, "vendor", "tree-sitter-typescript", "tsx")
    php_grammar_loc = os.path.join(tree_sitter_dir, "vendor", "tree-sitter-php")
    c_sharp_grammar_loc = os.path.join(tree_sitter_dir, "vendor", "tree-sitter-c-sharp")
    c_grammar_loc = os.path.join(tree_sitter_dir, "vendor", "tree-sitter-c")
    bash_grammar_loc = os.path.join(tree_sitter_dir, "vendor", "tree-sitter-bash")
    rust_grammar_loc = os.path.join(tree_sitter_dir, "vendor", "tree-sitter-rust")
    # location for library
    bin_loc = get_tree_sitter_so()
    # build everything
    Language.build_library(
        # Store the library in the `bin_loc`
        bin_loc,
        # Include languages
        [
            javascript_grammar_loc,
            python_grammar_loc,
            java_grammar_loc,
            go_grammar_loc,
            cpp_grammar_loc,
            ruby_grammar_loc,
            typescript_grammar_loc,
            tsx_grammar_loc,
            php_grammar_loc,
            c_sharp_grammar_loc,
            c_grammar_loc,
            bash_grammar_loc,
            rust_grammar_loc
        ]
    )
    print("Parser successfully initialized.")


def get_parser(lang: str) -> Parser:
    """
    Initialize parser for a specific language.
    :param lang: language to use.
    :return: parser.
    """
    global PARSERS
    if lang not in PARSERS:
        parser = Parser()
        parser.set_language(Language(get_tree_sitter_so(), lang))
        PARSERS[lang] = parser
    else:
        parser = PARSERS[lang]
    return parser
