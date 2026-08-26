"""Focused fixtures for synthetic-world diagnostic semantics."""

import unittest

from mayajaal.synthetic.diagnostics import graph_statistics


class GraphDiagnosticTests(unittest.TestCase):
    def test_bipartite_four_cycles_count_cross_type_identity_pairs(self) -> None:
        accounts = ("a", "b", "c")
        identity_accounts = {
            "device:d1": {"a", "b"},
            "address:addr1": {"a", "b"},
            "payment:p1": {"a", "b"},
            "ip:ip1": {"c"},
        }
        graph = graph_statistics(accounts, identity_accounts, set())
        # a and b share three identity nodes, so they form C(3, 2) K2,2s.
        self.assertEqual(graph["account_identity_four_cycle_count"], 3.0)

    def test_one_shared_identity_is_not_a_bipartite_four_cycle(self) -> None:
        graph = graph_statistics(("a", "b"), {"device:d1": {"a", "b"}}, set())
        self.assertEqual(graph["account_identity_four_cycle_count"], 0.0)


if __name__ == "__main__":
    _ = unittest.main()
