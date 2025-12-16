import re


def inject_headers(text):
    """
    Injects Markdown headers to force granularity for Audit files.
    - numeric starts (1. , 2. ) -> ##
    - squares (□) -> ###
    - circles (○) -> ####
    - dashes (-) -> #####
    """
    lines = text.split("\n")
    new_lines = []

    # Pre-compile regex
    # Case 1: "1. ", "2. " (exclude dates like 2024.)
    # Look for start of line, digit, dot, space.
    # But be careful of dates. Usually dates are 2024. 5.
    # Let's target strictly "Digit. " or "Digit " at start.
    # The Alio example had "1 감사 개요".

    pat_num_dot = re.compile(r"^(\d+)\.\s+(.*)")
    pat_num_space = re.compile(r"^(\d+)\s+(.*)")
    pat_square = re.compile(r"^[□■]\s*(.*)")
    pat_circle = re.compile(r"^[○●]\s*(.*)")
    # pat_dash = re.compile(r'^-\s+(.*)')

    for line in lines:
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            new_lines.append(line)
            continue

        # [Safety] Header candidate must be short (< 100 chars)
        # If it's a long paragraph starting with "1. ", it's not a header.
        if len(stripped) > 100:
            new_lines.append(line)
            continue

        # Check patterns
        # 1. "1. Title" -> "## 1. Title"
        m = pat_num_dot.match(line)
        if m:
            # Check if it looks like a year (e.g. 2024.) - naive check
            if int(m.group(1)) > 1900 and int(m.group(1)) < 2100:
                new_lines.append(line)
            else:
                new_lines.append(f"## {line}")
            continue

        # 2. "1 Title" -> "## 1 Title" (Alio style)
        m = pat_num_space.match(line)
        if m:
            # Check year
            if int(m.group(1)) > 1900 and int(m.group(1)) < 2100:
                new_lines.append(line)
            else:
                new_lines.append(f"## {line}")
            continue

        # 3. Square
        if pat_square.match(line):
            new_lines.append(f"### {line}")
            continue

        # 4. Circle
        if pat_circle.match(line):
            new_lines.append(f"#### {line}")
            continue

        new_lines.append(line)

    return "\n".join(new_lines)


# Test with Alio file "1. 목 적" pattern
file_path = "00_data/parsed_data/alio_local/C0270/2024041202799610-00.md"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

injected = inject_headers(text)

# Print first 2000 chars to see headers
print(injected[:2000])
