import sys
import re
from pathlib import Path

FILES = [r"content/about.md", r"config.toml"] 

def update_config_toml(filepath, new_url):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    updated_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Detect start of a social block
        if line.strip() == '[[params.social]]':
            block = [line]
            i += 1
            while i < len(lines) and not lines[i].startswith('['):
                block.append(lines[i])
                i += 1

            # Modify only if name == "Resume"
            if any(re.search(r'name\s*=\s*"Resume"', b) for b in block):
                new_block = []
                for b in block:
                    if b.strip().startswith("url"):
                        new_block.append(f'url = "{new_url}"\n')
                    else:
                        new_block.append(b)
                updated_lines.extend(new_block)
            else:
                updated_lines.extend(block)
        else:
            updated_lines.append(line)
            i += 1

    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(updated_lines)




def update_markdown_file(filepath, new_url):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    updated_lines = []
    for line in lines:
        # replace [Resume](...)
        line = re.sub(r'\[Resume\]\([^)]+\)', f'[Resume]({new_url})', line)
        updated_lines.append(line)

    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(updated_lines)


def main():
    if len(sys.argv) != 2:
        print("Usage: python change_resume_link.py NEW_URL")
        sys.exit(1)

    new_url = sys.argv[1]

    for file in FILES:
        path = Path(file)
        if path.exists():
            print(f"Change Resume URL in {file}")
            if file.endswith(".md"):
                update_markdown_file(path, new_url)
            elif file.endswith(".toml"):
                update_config_toml(path, new_url)
        else:
            print(f"Markdown file not found: {file}")

    print("All done!!!!!")

if __name__ == "__main__":
    main()