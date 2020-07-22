from typing import Optional, Set, Generator

import tree_sitter


def traverse_down_with_cursor(cursor: tree_sitter.TreeCursor, types: Optional[Set[str]] = None) \
        -> Generator[tree_sitter.Node, None, None]:
    """
    Traverse down AST tree for given cursor in DFS-order and yield nodes with given set of types. All node types are
    extracted if types is None.

    :param cursor: cursor to extract entities from
    :param types: types used for extraction
    :return: generator of nodes
    """
    has_next_child = True
    has_next_sibling = True
    has_parent_node = True

    while has_next_child or has_next_sibling:
        if types is None or cursor.node.type in types:
            yield cursor.node

        # Traverse down
        has_next_child = cursor.goto_first_child()

        # If leaf node is met try traverse right or find parent node where we can traverse right
        if not has_next_child:
            has_next_sibling = cursor.goto_next_sibling()

            while not has_next_sibling and has_parent_node:
                has_parent_node = cursor.goto_parent()
                has_next_sibling = cursor.goto_next_sibling()
