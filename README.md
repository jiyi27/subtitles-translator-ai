# subtitles-translator-ai

## Installation

Add the OpenAI API key to your .bashrc file in the root of your home folder (.zshrc if you use zsh).

```bash
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```

Then run the following command to install the required packages.

```bash
pip install -r requirements.txt
```
## Usage

```bash
python3 translator.py -i <input_srt_file> -o <output_srt_file>
```