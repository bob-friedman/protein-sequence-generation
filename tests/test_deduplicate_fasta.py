import unittest
import subprocess
import tempfile
import sys
import os

# Determine the project root directory assuming tests are in a 'tests' subdirectory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class TestDeduplicateFastaScript(unittest.TestCase):

    def run_script(self, input_content):
        """Helper function to run the deduplicate_fasta.py script with given input content."""
        try:
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.fasta', dir=PROJECT_ROOT) as tmp_input_file:
                tmp_input_file.write(input_content)
                tmp_input_file.flush()
                tmp_input_file_name = tmp_input_file.name
            
            # Using sys.executable to ensure the correct Python interpreter is used.
            # Adding PROJECT_ROOT to PYTHONPATH for the subprocess to find the 'scripts' module.
            env = os.environ.copy()
            python_path = env.get('PYTHONPATH', '')
            env['PYTHONPATH'] = f"{PROJECT_ROOT}{os.pathsep}{python_path}"

            process = subprocess.run(
                [sys.executable, os.path.join(PROJECT_ROOT, 'scripts', 'deduplicate_fasta.py'), tmp_input_file_name],
                capture_output=True,
                text=True,
                env=env
            )
            return process
        finally:
            if os.path.exists(tmp_input_file_name):
                os.remove(tmp_input_file_name)

    def test_duplicate_sequences(self):
        input_fasta = """>seq1
ABC
>seq2
DEF
>seq1_duplicate
ABC
>seq3
GHI
"""
        expected_output = """>seq1
ABC
>seq2
DEF
>seq3
GHI
"""
        result = self.run_script(input_fasta)
        self.assertEqual(result.returncode, 0, f"Script failed with stderr: {result.stderr}")
        self.assertEqual(result.stdout.strip(), expected_output.strip())

    def test_duplicates_after_whitespace_removal(self):
        input_fasta = """>seqA
X Y Z
>seqB
XYZ
>seqC
PQR
"""
        # The script should print the original sequence associated with the first seen hash
        expected_output = """>seqA
X Y Z
>seqC
PQR
"""
        result = self.run_script(input_fasta)
        self.assertEqual(result.returncode, 0, f"Script failed with stderr: {result.stderr}")
        self.assertEqual(result.stdout.strip(), expected_output.strip())

    def test_multiline_sequences_with_duplicates(self):
        input_fasta = """>multi1
ABC
DEF
>multi2
GHI
JKL
>multi1_dup
AB
CD EF
>unique
MNO
"""
        # parse_fasta joins lines, so output sequences are single lines.
        expected_output = """>multi1
ABCDEF
>multi2
GHIJKL
>unique
MNO
"""
        result = self.run_script(input_fasta)
        self.assertEqual(result.returncode, 0, f"Script failed with stderr: {result.stderr}")
        self.assertEqual(result.stdout.strip(), expected_output.strip())

    def test_empty_fasta_file(self):
        input_fasta = ""
        expected_output = ""
        result = self.run_script(input_fasta)
        self.assertEqual(result.returncode, 0, f"Script failed with stderr: {result.stderr}")
        self.assertEqual(result.stdout.strip(), expected_output.strip())

    def test_fasta_with_only_headers(self):
        input_fasta = """>header1
>header2
"""
        expected_output = ""
        result = self.run_script(input_fasta)
        # The script exits with 0 if file is found, even if no sequences are processed.
        self.assertEqual(result.returncode, 0, f"Script failed with stderr: {result.stderr}")
        self.assertEqual(result.stdout.strip(), expected_output.strip())

    def test_unique_sequences_duplicated_headers(self):
        input_fasta = """>header
SEQ1
>header
SEQ2
"""
        expected_output = """>header
SEQ1
>header
SEQ2
"""
        result = self.run_script(input_fasta)
        self.assertEqual(result.returncode, 0, f"Script failed with stderr: {result.stderr}")
        self.assertEqual(result.stdout.strip(), expected_output.strip())
    
    def test_file_not_found(self):
        # Test non-existent file directly without helper
        env = os.environ.copy()
        python_path = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = f"{PROJECT_ROOT}{os.pathsep}{python_path}"
        
        process = subprocess.run(
            [sys.executable, os.path.join(PROJECT_ROOT, 'scripts', 'deduplicate_fasta.py'), 'non_existent_file.fasta'],
            capture_output=True,
            text=True,
            env=env
        )
        self.assertNotEqual(process.returncode, 0) # Expecting non-zero exit code
        self.assertIn("Error: File not found", process.stderr)


if __name__ == '__main__':
    unittest.main()
