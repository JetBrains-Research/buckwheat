"""
The downloading of Enry
"""
import json
import logging
import os
import platform
import subprocess
from typing import Dict, List
import urllib.request

DOWNLOAD_URLS = {
    "Linux":
        "https://github.com/go-enry/enry/releases/download/v1.0.0/enry-v1.0.0-linux-amd64.tar.gz",
    "Darwin":
        "https://github.com/go-enry/enry/releases/download/v1.0.0/enry-v1.0.0-darwin-amd64.tar.gz"
}

FILENAMES = {
    "Linux": "enry.tar.gz",
    "Darwin": "enry.tar.gz"
}


def identify_system() -> str:
    """
    Get the system name. Supported systems are Linux and Darwin (MacOS).
    :return: system name.
    """
    system = platform.system()
    if system not in ["Linux", "Darwin"]:
        raise ValueError(f"Unsupported system {system}")
    return system


def get_enry_dir() -> str:
    """
    Get the directory with Enry.
    :return: absolute path.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "build"))


def get_enry() -> str:
    """
    Get the path to the Enry binary.
    :return: absolute path.
    """
    return os.path.abspath(os.path.join(get_enry_dir(), "enry"))


def main() -> None:
    """
    Download Enry.
    :return: None.
    """
    system = identify_system()
    url = DOWNLOAD_URLS[system]
    filename = FILENAMES[system]
    if not os.path.exists(os.path.abspath(os.path.join(get_enry_dir(), filename))):
        try:
            urllib.request.urlretrieve(url, os.path.abspath(os.path.join(get_enry_dir(),
                                                                         filename)))
        except Exception as e:
            logging.error("Failed to download language recognizer. {type}: {error}."
                          .format(type=type(e).__name__, error=e))
    if not os.path.exists(get_enry()):
        try:
            os.system("tar -xzf {tar} -C {directory}"
                      .format(tar=os.path.abspath(os.path.join(get_enry_dir(), filename)),
                              directory=get_enry_dir()))
        except Exception as e:
            logging.error("Failed to unpack language recognizer. {type}: {error}."
                          .format(type=type(e).__name__, error=e))
    logging.info("Language recognizer successfully initialized.")


def recognize_languages_dir(directory: str) -> Dict[str, List[str]]:
    """
    Recognize the languages in the directory using Enry and return a dictionary.
    {language1: [files], language2: [files], ...}.
    :param directory: the path to the directory.
    :return: dictionary {language1: [files], language2: [files], ...}
    """
    enry = get_enry()
    args = [enry, "-json", directory]
    res = subprocess.check_output(args)
    return json.loads(res)


def recognize_language_file(file_path: str) -> Dict[str, str]:
    """
    Recognize the language of a file.
    :param file_path: directory location to classify.
    :return: dictionary `{"filename":name,"language":lang,"lines":n_lines,"mime":mime,"total_lines":n_total_lines,
                         "type":type,"vendored":bool}`
    """
    if not os.path.isfile(file_path):
        raise ValueError("Expected path to file path but got '%s'" % file_path)
    enry = get_enry()
    args = [enry, "-json", file_path]
    res = subprocess.check_output(args)
    return json.loads(res)
