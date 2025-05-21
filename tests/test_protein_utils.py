import unittest
import sys
import os

# Add the parent directory to the Python path to allow importing protein_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from protein_utils import format_sequence_with_spaces

class TestFormatSequenceWithSpaces(unittest.TestCase):

    def test_empty_string(self):
        self.assertEqual(format_sequence_with_spaces(""), "")

    def test_simple_uppercase_sequence(self):
        self.assertEqual(format_sequence_with_spaces("MFVFL"), "M F V F L ")

    def test_mixed_case_sequence(self):
        self.assertEqual(format_sequence_with_spaces("MfVfL"), "M f V f L ")

    def test_endoftext_at_ends(self):
        self.assertEqual(format_sequence_with_spaces("<|endoftext|>SEQUENCE<|endoftext|>"), "S E Q U E N C E ")

    def test_endoftext_in_middle(self):
        self.assertEqual(format_sequence_with_spaces("SEQ<|endoftext|>UENCE"), "S E Q U E N C E ")

    def test_newline_characters(self):
        self.assertEqual(format_sequence_with_spaces("SEQ\nUENCE"), "S E Q U E N C E ")

    def test_carriage_return_characters(self):
        self.assertEqual(format_sequence_with_spaces("SEQ\rUENCE"), "S E Q U E N C E ")

    def test_sequence_with_numbers_and_symbols(self):
        self.assertEqual(format_sequence_with_spaces("SEQ123*UENCE"), "S E Q 123*U E N C E ")

    def test_sequence_with_leading_trailing_spaces_around_letters(self):
        # The current function preserves leading/trailing spaces if they are outside the letters.
        # If ' M F V F L ' is input, the space is not an alphabet, so it's kept.
        # Then M, F, V, F, L get spaces after them.
        self.assertEqual(format_sequence_with_spaces(" MFVFL "), " M F V F L  ") # Corrected expected output

    def test_string_only_endoftext(self):
        self.assertEqual(format_sequence_with_spaces("<|endoftext|>"), "")

    def test_multiple_unwanted_characters(self):
        self.assertEqual(format_sequence_with_spaces("<|endoftext|>SEQ\n<|endoftext|>UENCE\rANOTHER<|endoftext|>"), "S E Q U E N C E A N O T H E R ")

if __name__ == '__main__':
    unittest.main()
