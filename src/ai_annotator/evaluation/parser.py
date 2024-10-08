import re
import json

def parse_first_int(strings: list[str], bos_split_token: str = None, eos_split_token: str = None, default_value: int = None) -> list[int]:
    """
    Extracts the first integer from each string in a list and returns a list of these integers.

    Args:
        strings: A list of strings to parse.
        split_token: A substring to split each string at before searching for an integer.
        default_value: A default integer to return if no integer is found in a string.
    """
    parsed_integers: list[int] = []
    for string in strings:
        if bos_split_token:
            string = string.split(bos_split_token)[-1]
        if eos_split_token:
            string = string.split(eos_split_token)[0]
        try:
            number = re.search(r'\d+', string).group()
            parsed_integers.append(int(number))
        except AttributeError:
            parsed_integers.append(default_value)

    return parsed_integers
    

def parse_list(strings: list[str], bos_split_token: str = None, eos_split_token: str = None, delimiter: str = ",", default_value: list[str] = []) -> list[list[str]]:
    """
    Splits each string in a list into sublists of strings based on a delimiter and returns a list of these sublists.

    Args:
        strings: A list of strings to parse.
        bos_split_token: If provided, the string will be split at this token, and only the part after this token will be processed.
        eos_split_token: If provided, the string will be split at this token, and only the part before this token will be processed.
        delimiter: A substring used to split each string into sublists.
        default_value: A default list of strings to return if an error occurs during parsing.
    """

    parsed_lists: list[list[str]] = []
    for string in strings:
        if bos_split_token:
            string = string.split(bos_split_token)[-1]
        if eos_split_token:
            string = string.split(eos_split_token)[0]

        try:
            split_strings: list[str] = string.split(delimiter)
            parsed_lists.append([s.strip() for s in split_strings])
        except AttributeError:
            parsed_lists.append(default_value)

    return parsed_lists


def parse_json(strings: list[str], default_value: dict = {}) -> list[dict]:
    """
    Parses each string in a list as JSON and returns a list of the parsed JSON objects.

    Args:
        strings: A list of strings to parse.
        default_value: A default dictionary to return if an error occurs during parsing.

    Note:
    - Not tested!
    """
    
    parsed_jsons: list[dict] = []
    for string in strings:
        try:
            parsed_json = json.loads(string)
            parsed_jsons.append(parsed_json)
        except json.JSONDecodeError:
            parsed_jsons.append(default_value)

    return parsed_jsons