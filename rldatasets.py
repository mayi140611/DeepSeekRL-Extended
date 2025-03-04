"""
Hold all data sets 

"""

import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset
from abc import ABC, abstractmethod
from typing import Tuple, Any



class DataLoader(ABC):
    """
    Abstract base class for data loaders.
    
    This class defines the interface that all dataset loaders should implement.
    Specific dataset loaders should inherit from this class and implement the
    required methods.
    
    Attributes:
        random (bool): If True, returns items randomly; if False, returns sequentially
        current_index (int): Current position for sequential access
    """
    
    def __init__(self, random: bool = False) -> None:
        self.random = random
        self.current_index = 0
        
    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of items in the dataset."""
        pass
        
    @abstractmethod
    def __iter__(self) -> 'DataLoader':
        """Return self as iterator."""
        return self
        
    @abstractmethod
    def __next__(self) -> Any:
        """Return the next item(s) in the dataset."""
        pass


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()



SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

class WmtEng2ZhLoader(DataLoader):
    """
    A loader class that provides iteration over WmtEng2Zh.
    
    This class implements both sequential and random access to math problems through
    standard Python iterator protocols. It can be used to iterate over problems either
    in order or randomly, making it suitable for both training and evaluation.
    
    Attributes:
        questions (List[str]): List of math question strings
        answers (List[str]): List of corresponding answer strings
        random (bool): If True, returns problems randomly; if False, returns sequentially
        current_index (int): Current position in the lists for sequential access
    """
    
    def __init__(self, questions: list[str], answers: list[str], random: bool = False) -> None:
        super().__init__(random)
        self.questions = questions
        self.answers = answers
        # 把给定的英语直译为汉语，注意尽量按照字面意思直接翻译，翻译结果要保持自然连贯，不要衍生其它内容。
        self.pre_prompt = """Translate the given English into Chinese, pay attention to the literal translation as much as possible, The translation result should be kept natural and coherent, and do not derive other content.
            And put your thinking process inside <reasoning></reasoning>tags and your final translation inside <answer></answer> tags, like this:
            
            <reasoning>
            Your step-by-step thinking process here
            </reasoning>
            <answer>
            Your final translation here
            </answer>

            NOTE: All of your returned text should either be in the <reasoning> or <answer> tags - no text outside! 

            Text: """
        self.system_prompt = self.pre_prompt#SYSTEM_PROMPT
        
    def __len__(self) -> int:
        return len(self.questions)
        
    def __iter__(self) -> 'WmtEng2ZhLoader':
        return self
        
    def __next__(self) -> tuple[str, str]:
        if self.current_index >= len(self.questions):
            raise StopIteration
        
        if self.random:
            idx = random.randint(0, len(self.questions) - 1)
        else:
            idx = self.current_index
            self.current_index += 1
            
        return self.questions[idx], self.answers[idx]

    def reset(self):
        self.current_index = 0 


def build_wmteng2zh_dataloaders() -> Tuple[WmtEng2ZhLoader, WmtEng2ZhLoader]: 
    dir1 = '/home/jupyter/ollama_models/blob/mm/0myprojects220424/a-250220_0331-mine-复现r1/output/'
    with open(dir1 + 'newsdev2021.ha-en.en') as f\
        , open(dir1 + 'newsdev2021.ha-en.zh') as f1:
        data_en = f.read().split('\n')
        data_pcm = f1.read().split('\n')
        print(len(data_en), len(data_pcm))

    questions = data_en
    parsed_answers = data_pcm

    # Randomly split into train/test sets
    total_samples = len(questions)
    test_size = int(total_samples * 0.019)  # 10% for test set
    
    # Generate random indices for test set
    test_indices = random.sample(range(total_samples), test_size)
    test_indices_set = set(test_indices)
    
    # Convert to numpy arrays for easier indexing
    questions = np.array(questions)
    parsed_answers = np.array(parsed_answers)
    
    # Create boolean mask for test indices
    test_mask = np.zeros(total_samples, dtype=bool)
    test_mask[list(test_indices_set)] = True
    
    # Split using boolean indexing
    test_questions = questions[test_mask]
    test_answers = parsed_answers[test_mask]
    train_questions = questions[~test_mask] 
    train_answers = parsed_answers[~test_mask]

    # Setup data loaders 
    trainloader = WmtEng2ZhLoader(train_questions.tolist(), train_answers.tolist())
    testloader = WmtEng2ZhLoader(test_questions.tolist(), test_answers.tolist())
    
    return trainloader, testloader

class Eng2PidginLoader(DataLoader):
    """
    A loader class that provides iteration over Eng2Pidgin.
    
    This class implements both sequential and random access to math problems through
    standard Python iterator protocols. It can be used to iterate over problems either
    in order or randomly, making it suitable for both training and evaluation.
    
    Attributes:
        questions (List[str]): List of math question strings
        answers (List[str]): List of corresponding answer strings
        random (bool): If True, returns problems randomly; if False, returns sequentially
        current_index (int): Current position in the lists for sequential access
    """
    
    def __init__(self, questions: list[str], answers: list[str], random: bool = False) -> None:
        super().__init__(random)
        self.questions = questions
        self.answers = answers
        self.pre_prompt = """You need to translate the given English as Nigerian Pidgin. You should think carefully about the text, and translate it to Nigerian Pidgin, then provide your translation.
            It is very important that you put your thinking process inside <reasoning> tags and your final translation inside <answer> tags, like this:

            
            <reasoning>
            Your step-by-step thinking process here
            </reasoning>
            <answer>
            Your final translation here
            </answer>

            All of your returned text should either be in the <reasoning> or <answer> tags - no text outside! Start each answer by immediately starting with <reasoning>. 
            It is is extremely important you answer in this way - do not put any information or text outside of these tags!

            Text: """
        self.system_prompt = self.pre_prompt#SYSTEM_PROMPT
        
    def __len__(self) -> int:
        return len(self.questions)
        
    def __iter__(self) -> 'Eng2PidginLoader':
        return self
        
    def __next__(self) -> tuple[str, str]:
        if self.current_index >= len(self.questions):
            raise StopIteration
        
        if self.random:
            idx = random.randint(0, len(self.questions) - 1)
        else:
            idx = self.current_index
            self.current_index += 1
            
        return self.questions[idx], self.answers[idx]

    def reset(self):
        self.current_index = 0 


def build_eng2pidgin_dataloaders() -> Tuple[Eng2PidginLoader, Eng2PidginLoader]: 
    dir1 = '/home/jupyter/ollama_models/blob/mm/0myprojects220424/a-250121_0630-cy-input_method/'
    with open(dir1 + 'dataset/naija_treebank/train.en') as f\
        , open(dir1 + 'dataset/naija_treebank/train.pcm') as f1:
        data_en = f.read().split('\n')
        data_pcm = f1.read().split('\n')
        print(len(data_en), len(data_pcm))

    questions = data_en
    parsed_answers = data_pcm

    # Randomly split into train/test sets
    total_samples = len(questions)
    test_size = int(total_samples * 0.0013)  # 10% for test set
    
    # Generate random indices for test set
    test_indices = random.sample(range(total_samples), test_size)
    test_indices_set = set(test_indices)
    
    # Convert to numpy arrays for easier indexing
    questions = np.array(questions)
    parsed_answers = np.array(parsed_answers)
    
    # Create boolean mask for test indices
    test_mask = np.zeros(total_samples, dtype=bool)
    test_mask[list(test_indices_set)] = True
    
    # Split using boolean indexing
    test_questions = questions[test_mask]
    test_answers = parsed_answers[test_mask]
    train_questions = questions[~test_mask] 
    train_answers = parsed_answers[~test_mask]

    # Setup data loaders 
    trainloader = Eng2PidginLoader(train_questions.tolist(), train_answers.tolist())
    testloader = Eng2PidginLoader(test_questions.tolist(), test_answers.tolist())
    
    return trainloader, testloader

class GSM8KLoader(DataLoader):
    """
    A loader class that provides iteration over GSM8K math problems.
    
    This class implements both sequential and random access to math problems through
    standard Python iterator protocols. It can be used to iterate over problems either
    in order or randomly, making it suitable for both training and evaluation.
    
    Attributes:
        questions (List[str]): List of math question strings
        answers (List[str]): List of corresponding answer strings
        random (bool): If True, returns problems randomly; if False, returns sequentially
        current_index (int): Current position in the lists for sequential access
    """
    
    def __init__(self, questions: list[str], answers: list[str], random: bool = False) -> None:
        super().__init__(random)
        self.questions = questions
        self.answers = answers
        self.pre_prompt = """You will be given a question that involves reasoning. You should reason carefully about the question, then provide your answer.
            It is very important that you put your reasoning process inside <reasoning> tags and your final answer inside <answer> tags, like this:

            
            <reasoning>
            Your step-by-step reasoning process here
            </reasoning>
            <answer>
            Your final answer here
            </answer>

            All of your returned text should either be in the <reasoning> or <answer> tags - no text outside! Start each answer by immediately starting with <reasoning>. 
            It is is extremely important you answer in this way - do not put any information or text outside of these tags!

            Question: """
        self.system_prompt = SYSTEM_PROMPT
        
    def __len__(self) -> int:
        return len(self.questions)
        
    def __iter__(self) -> 'GSM8KLoader':
        return self
        
    def __next__(self) -> tuple[str, str]:
        if self.current_index >= len(self.questions):
            raise StopIteration
        
        if self.random:
            idx = random.randint(0, len(self.questions) - 1)
        else:
            idx = self.current_index
            self.current_index += 1
            
        return self.questions[idx], self.answers[idx]

    def reset(self):
        self.current_index = 0 


def build_gsm8k_dataloaders() -> Tuple[GSM8KLoader, GSM8KLoader]: 
    ds = load_dataset('openai/gsm8k', 'main')
    loaders = []
    for tt in "train test".split():
        data = ds[tt]
        questions = []
        parsed_answers = [] 
        for i in tqdm(range(len(data)), desc="Processing"):
            # Try to get answer - if is None dont use this sample 
            ans = extract_hash_answer(data[i]['answer'])
            if ans is None: 
                continue 
            else:
                questions.append(data[i]['question'])
                parsed_answers.append(ans)

        # # Randomly split into train/test sets
        # total_samples = len(questions)
        # test_size = int(total_samples * 0.01)  # 10% for test set
        
        # # Generate random indices for test set
        # test_indices = random.sample(range(total_samples), test_size)
        # test_indices_set = set(test_indices)
        
        # # Convert to numpy arrays for easier indexing
        # questions = np.array(questions)
        # parsed_answers = np.array(parsed_answers)
        
        # # Create boolean mask for test indices
        # test_mask = np.zeros(total_samples, dtype=bool)
        # test_mask[list(test_indices_set)] = True
        
        # # Split using boolean indexing
        # test_questions = questions[test_mask]
        # test_answers = parsed_answers[test_mask]
        # train_questions = questions[~test_mask] 
        # train_answers = parsed_answers[~test_mask]

        # Setup data loaders 
        # trainloader = GSM8KLoader(train_questions.tolist(), train_answers.tolist())
        # testloader = GSM8KLoader(test_questions.tolist(), test_answers.tolist())
        if tt == 'test':
            len1 = 19
            loaders.append(GSM8KLoader(questions[:len1], parsed_answers[:len1]))
        else:
            loaders.append(GSM8KLoader(questions, parsed_answers))
    return tuple(loaders) #trainloader, testloader


def get_dataloaders(dataset_name: str) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to get train and test data loaders for a specified dataset.
    
    Args:
        dataset_name (str): Name of the dataset to load ('gsm8k' currently supported)
        
    Returns:
        Tuple[DataLoader, DataLoader]: Train and test data loaders
        
    Raises:
        ValueError: If dataset_name is not supported
    """
    if dataset_name.lower() == 'gsm8k':
        return build_gsm8k_dataloaders()
    elif dataset_name.lower() == 'eng2pidgen':
        return build_eng2pidgin_dataloaders()
    elif dataset_name.lower() == 'wmteng2zh':
        return build_wmteng2zh_dataloaders()
    else:
        raise ValueError(f"Dataset {dataset_name} not supported. Currently only 'gsm8k' is available.")


if __name__ == "__main__": 
    trainloader, testloader = get_dataloaders('gsm8k')