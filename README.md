# Sourcer Code Identifiers
A multi-language tokenizer for extracting identifiers (or, theoretically, anything else) from source code.
## How to use
1. The project uses [tree-sitter](https://tree-sitter.github.io/) and its grammars as submodules, so clone the repo recursively.
2. Create an input file with a list of links to GitHub repositories.
3. Run from the command line with `python3 -m topic_dynamics.run` and the following arguments:
    - `-i`: a path to the input file;
    - `-o`: a path to the output directory;
    - `-b`: the size of the batch of projects that will be saved together (by default 100). 

For every batch, two files will be created:
- `docword`: for every repository, all of its subtokens are listed as `id:count`, one repository per line, in descending order of counts. The ids are the same for the entire batch.
- `vocab`: all unique subtokens are listed as `id;subtoken`, one subtoken per line, in ascending order of ids.

## How it works
After the target project is downloaded, it is processed in three main steps:
1. **Language recognition**. Firstly, the languages of the project are recognized with [enry](https://github.com/src-d/enry). This operation returns a dictionary with languages as keys and corresponding lists of files as values. Only the files in supported languages are passed on to the next step (see the full list below).
2. **Parsing**. Every file is parsed with one of the two parsers. The most popular languages are parsed with [tree-sitter](https://tree-sitter.github.io/), and the languages that do not yet have _tree-sitter_ grammar are parsed with [pygments](https://pygments.org/). At this point, identifiers are extracted and every identifier is passed on to the next step.
3. **Subtokenizing**. Every identifier is split into subtokens by camelCase and snake_case.

The counters of subtokens are aggregated for projects and saved to file.

## Advanced use

Every step of the pipeline can be modified:
1. Languages can be added by modifying `SUPPORTED_LANGUAGES` in `parsing.py`.
2. The tool can extract not only identifiers, but anything that is detected by either _tree-sitter_ or _pygments_. This can be done my modifying `NODE_TYPES` in `TreeSitterParser` class and `TYPES` in `PygmentsParser` class.
3. Subtokenization can be modified in `subtokenizing.py`. The tokens can be connected together, stemmed, filtered by length, etc.

## Supported languages
Currently, the following languages are supported: C, C#, C++, Go, Java, JavaScript, Kotlin, PHP, Python, Ruby, Rust, Scala, Shell, Swift, and TypeScript.
