def format_sequence_with_spaces(sequence_text: str) -> str:
  """
  Formats a protein sequence string by removing specific unwanted characters and
  adding a space after each alphabet letter.

  Args:
    sequence_text: The input protein sequence string.

  Returns:
    The processed string with spaces after alphabet letters and unwanted
    characters removed.
  """
  # Remove unwanted characters
  sequence_text = sequence_text.replace("\n", "")
  sequence_text = sequence_text.replace("\r", "")
  sequence_text = sequence_text.replace("<|endoftext|>", "")

  processed_sequence = []
  for char in sequence_text:
    if 'a' <= char.lower() <= 'z':  # Check if the character is an alphabet letter
      processed_sequence.append(char)
      processed_sequence.append(" ")
    else:
      processed_sequence.append(char)
  
  return "".join(processed_sequence)
