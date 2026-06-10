import re

import numpy as np


def simplify_dot_names(input_file, output_file):
    # Matches quoted full names like "A.B.C.ClassName"
    pattern = re.compile(r'"([^"]+)"')

    def shorten(match):
        full_name = match.group(1)
        short_name = full_name.split('.')[-1]
        return f'"{short_name}"'

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace all quoted names
    new_content = pattern.sub(shorten, content)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(new_content)

if __name__ == "__main__":
    print(np.exp(100))
    # simplify_dot_names("packages_my_project.dot", "output.dot")