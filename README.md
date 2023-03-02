# chatbot-DiabloGPT

## Overview
Chatbot-DiabloGPT is a simple chatbot built using `transformers`, `torch` and `gradio`. It can change the pre-trained model

## Requirements
- [Python](https://www.python.org/downloads) `3.x` (I've tested version 3.10.9)
- [Gradio](https://gradio.app) version `3.19.1`
- [Transformers](https://pypi.org/project/transformers) version `4.22.2`
    * From version `4.23` has warning message: AutoModelForCasualLM "decoder-only architecture" warning, even after setting padding_side='left'   
- [Pytorch](https://pypi.org/project/torch) version `1.13.1`

All packages in the file `requirements.txt`. You should use a virtual environment like `miniconda`.

## Installation

1. Clone the repository
2. Navigate to the project directory
3. Run `pip install -r requirements.txt` to install the required dependencies


## Usage
1. Change the following line in `main.py` to run the chatbot publicly:
```python
# Public
block.launch(share=True)

# Local sharing: ip, port
block.launch(server_name="0.0.0.0", server_port=5050)

# Local dev: http://127.0.0.1:7860
block.launch(debug=True)
```

2. Run in Windows
```python
python main.py
```

3. Run in Linux
```python
python main.py

# Or run in background write log
nohup python -u main.py > nohup.txt &
```

4. Check the `nohup.txt` file or the `logs` folder for the chatbot's output.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.