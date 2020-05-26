[![JetBrains Research](https://jb.gg/badges/research.svg)](https://confluence.jetbrains.com/display/ALL/JetBrains+on+GitHub)
![Linux & MacOS build](https://github.com/JetBrains-Research/identifiers-extractor/workflows/Linux%20&%20MacOS%20CI/badge.svg)

# Source Code Identifiers
A multi-language tokenizer for extracting identifiers (or, theoretically, anything else) from source code.

The tool is already employed in [searching for similar repositories](https://github.com/JetBrains-Research/similar-repositories/) and [studying the dynamics of topics in code](https://github.com/areyde/topic-dynamics).

## How to use
The tool currently works on Linux and MacOS, correct versions of files will be downloaded automatically. 
1. The project uses [tree-sitter](https://tree-sitter.github.io/) and its grammars as submodules, so update them after cloning: 

    ```shell script
    git submodule update --init --recursive --depth 1
    ```
    
2. Install the required dependencies:
    
    ```shell script
    pip3 install cython
    pip3 install -r requirements.txt
    ```
    
3. Create an input file with a list of repositories. In the default mode, the list must contain links to GitHub, in the local mode (activated by passing the `-l` argument), the list must contain the paths to local directories.
4. Run from the command line with `python3 -m identifiers_extractor.run` and the following arguments:
    - `-i`: a path to the input file;
    - `-o`: a path to the output directory;
    - `-b`: the size of the batch of projects that will be saved together (by default 100);
    - `-l`: if passed, switches the tokenization into the local mode, where the input file must contain the paths to local directories.

For every batch, two files will be created:
- `docword`: for every repository, all of its subtokens are listed as `id:count`, one repository per line, in descending order of counts. The ids are the same for the entire batch.
- `vocab`: all unique subtokens are listed as `id;subtoken`, one subtoken per line, in ascending order of ids.

## How it works
After the target project is downloaded, it is processed in three main steps:
1. **Language recognition**. Firstly, the languages of the project are recognized with [enry](https://github.com/src-d/enry). This operation returns a dictionary with languages as keys and corresponding lists of files as values. Only the files in supported languages are passed on to the next step (see the full list below).
2. **Parsing**. Every file is parsed with one of the two parsers. The most popular languages are parsed with [tree-sitter](https://tree-sitter.github.io/), and the languages that do not yet have _tree-sitter_ grammar are parsed with [pygments](https://pygments.org/). At this point, identifiers are extracted and every identifier is passed on to the next step.
3. **Subtokenizing**. Every identifier is split into subtokens by camelCase and snake_case, small subtokens are connected to longer ones, and the subtokens are stemmed. In general, the preprocessing is carried out as described in [this paper](https://arxiv.org/abs/1704.00135).

The counters of subtokens are aggregated for projects and saved to file.

## Advanced use

Every step of the pipeline can be modified:
1. Languages can be added by modifying `SUPPORTED_LANGUAGES` in `parsing.py`.
2. The tool can extract not only identifiers, but anything that is detected by either _tree-sitter_ or _pygments_. This can be done my modifying `NODE_TYPES` in `TreeSitterParser` class and `TYPES` in `PygmentsParser` class.
3. Subtokenization can be modified in `subtokenizing.py`. The tokens can be connected together, stemmed, filtered by length, etc.

## Supported languages
Currently, the following languages are supported: C, C#, C++, Go, Haskell, Java, JavaScript, Kotlin, PHP, Python, Ruby, Rust, Scala, Shell, Swift, and TypeScript.
