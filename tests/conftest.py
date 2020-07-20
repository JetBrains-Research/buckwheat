import pytest

from buckwheat.parsing.utils import main as init_tree_sitter
from buckwheat.language_recognition.utils import main as init_enry


@pytest.fixture(scope="session", autouse=True)
def setup_external_tools():
    init_tree_sitter()
    init_enry()
