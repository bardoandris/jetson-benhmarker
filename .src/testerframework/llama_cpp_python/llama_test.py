from llama_cpp import Llama

class llama_tester:
    def __init__(self, model_path):
        self.llama = Llama(model_path=model_path)
        self.llama.