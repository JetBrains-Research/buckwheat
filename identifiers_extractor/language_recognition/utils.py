"""
The downloading of Enry
"""
import os
import platform
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
        urllib.request.urlretrieve(url,
                                   os.path.abspath(os.path.join(get_enry_dir(), filename)))
    if not os.path.exists(get_enry()):
        os.system("tar -xzf {tar} -C {directory}"
                  .format(tar=os.path.abspath(os.path.join(get_enry_dir(), filename)),
                          directory=get_enry_dir()))
    print("Enry successfully initialized.")
