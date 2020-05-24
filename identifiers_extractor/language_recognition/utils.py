"""
The downloading of Enry
"""
import os
import platform
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


def identify_system() -> str:
    """
    Get the system name. Supported systems are Linux, Windows, and Darwin (MacOS).
    :return: system name.
    """
    system = platform.system()
    if system not in ["Linux", "Windows", "Darwin"]:
        raise ValueError(f"Unsupported system {system}")
    return system


download_urls = {
    "Linux": "https://github.com/go-enry/enry/releases/download/v1.0.0/enry-v1.0.0-linux-amd64.tar.gz",
    "Windows": "https://github.com/go-enry/enry/releases/download/v1.0.0/enry-v1.0.0-windows-amd64.zip",
    "Darwin": "https://github.com/go-enry/enry/releases/download/v1.0.0/enry-v1.0.0-darwin-amd64.tar.gz"
}

filenames = {
    "Linux": "enry.tar.gz",
    "Windows": "enry.zip",
    "Darwin": "enry.tar.gz"
}


def main() -> None:
    """
    Download Enry.
    :return: None.
    """
    system = identify_system()
    url = download_urls[system]
    filename = filenames[system]
    if not os.path.exists(os.path.abspath(os.path.join(get_enry_dir(), filename))):
        urllib.request.urlretrieve(url,
                                   os.path.abspath(os.path.join(get_enry_dir(), filename)))
    if not os.path.exists(get_enry()):
        os.system("tar -xzf {tar} -C {directory}"
                  .format(tar=os.path.abspath(os.path.join(get_enry_dir(), filename)),
                          directory=get_enry_dir()))
    print("Enry successfully initialized.")
