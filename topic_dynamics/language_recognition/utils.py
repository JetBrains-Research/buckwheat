"""
The downloading of Enry
"""
import os
import urllib.request


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
    url = "https://github.com/go-enry/enry/releases/download/v1.0.0/enry-v1.0.0-linux-amd64.tar.gz"
    if not os.path.exists(os.path.abspath(os.path.join(get_enry_dir(), "enry.tar.gz"))):
        urllib.request.urlretrieve(url,
                                   os.path.abspath(os.path.join(get_enry_dir(), "enry.tar.gz")))
    if not os.path.exists(get_enry()):
        os.system("tar -xzf {tar} -C {directory}"
                  .format(tar=os.path.abspath(os.path.join(get_enry_dir(), "enry.tar.gz")),
                          directory=get_enry_dir()))
    print("Enry successfully initialized.")
