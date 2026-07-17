import unittest

from scripts.analyze_failure_transfer_heldout import one_sided_exact_mcnemar


class FailureTransferHeldoutTests(unittest.TestCase):
    def test_exact_one_sided_mcnemar(self) -> None:
        self.assertEqual(one_sided_exact_mcnemar(0, 0), 1.0)
        self.assertEqual(one_sided_exact_mcnemar(3, 0), 0.125)
        self.assertEqual(one_sided_exact_mcnemar(5, 0), 0.03125)

    def test_control_direction_is_not_significant(self) -> None:
        self.assertEqual(one_sided_exact_mcnemar(0, 5), 1.0)


if __name__ == "__main__":
    unittest.main()
