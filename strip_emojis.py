import os
import re

# Comprehensive emoji regex
# Includes emoticons, symbols, pictographs, transport, flags, ornaments, and variations
EMOJI_PATTERN = re.compile(
    r"["
    r"\U0001f600-\U0001f64f"
    r"\U0001f300-\U0001f5ff"
    r"\U0001f680-\U0001f6ff"
    r"\U0001f1e6-\U0001f1ff"
    r"\u2600-\u27bf"
    r"\ufe0f"
    r"\u200d"
    r"\U0001f900-\U0001f9ff"
    r"\U0001f004"
    r"\U0001f0cf"
    r"\u231a\u231b\u23e9-\u23ec\u23f0\u23f3"
    r"\u2b50"
    r"]+", 
    flags=re.UNICODE
)

def strip_emojis(text):
    return EMOJI_PATTERN.sub('', text)

extensions = ('.py', '.md', '.html', '.css', '.js', '.json')

for root, dirs, files in os.walk('.'):
    # Skip .git and __pycache__
    if '.git' in dirs:
        dirs.remove('.git')
    if '__pycache__' in dirs:
        dirs.remove('__pycache__')
        
    for file in files:
        if file.endswith(extensions):
            path = os.path.join(root, file)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                new_content = strip_emojis(content)
                
                if new_content != content:
                    print(f"Stripping emojis from {path}")
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
            except Exception as e:
                print(f"Could not process {path}: {e}")

print("Emoji stripping complete.")
