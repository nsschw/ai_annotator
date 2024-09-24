import re

def parse_first_int(strings: list[str], delimiter: str = None, default_value: int = None) -> list[int]:
    """
    Extracts the first integer from each string in a list and returns a list of these integers.

    Args:
        strings: A list of strings to parse.
        delimiter: A substring to split each string at before searching for an integer.
        default_value: A default integer to return if no integer is found in a string.
    """
    parsed_integers: list[int] = []
    for string in strings:
        if delimiter:
            string = string.split(delimiter)[1]
        try:
            number = re.search(r'\d+', string).group()
            parsed_integers.append(int(number))
        except AttributeError:
            parsed_integers.append(default_value)

    return parsed_integers
    


    



