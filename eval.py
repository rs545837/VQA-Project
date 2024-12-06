from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

import nltk
from pywsd.lesk import simple_lesk
import numpy as np
import json
from PIL import Image
from tqdm import tqdm 

class SentenceSimilarity:
    
    def __init__(self):
        self.word_order = False
        
    
    def identifyWordsForComparison(self, sentence):
        #Taking out Noun and Verb for comparison word based
        tokens = nltk.word_tokenize(sentence)        
        pos = nltk.pos_tag(tokens)
        pos = [p for p in pos if p[1].startswith('N') or p[1].startswith('V')]     
        return pos
    
    def wordSenseDisambiguation(self, sentence):
        # removing the disambiguity by getting the context
        pos = self.identifyWordsForComparison(sentence)
        sense = []
        for p in pos:
            sense.append(simple_lesk(sentence, p[0], pos=p[1][0].lower()))
        return set(sense)
    
    def getSimilarity(self, arr1, arr2, vector_len):
        #cross multilping all domains 
        vector = [0.0] * vector_len
        count = 0
        for i,a1 in enumerate(arr1):
            all_similarityIndex=[]
            for a2 in arr2:
                similarity = a1.wup_similarity(a2)
                if similarity != None:
                    all_similarityIndex.append(similarity)
                else:
                    all_similarityIndex.append(0.0)
            all_similarityIndex = sorted(all_similarityIndex, reverse = True)
            vector[i]=all_similarityIndex[0]
            if vector[i] >= 0.804:
                count +=1
        return vector, count        

        
    def shortestPathDistance(self, sense1, sense2):
        #getting the shortest path to get the similarity
        if len(sense1) >= len(sense2):
            grt_Sense = len(sense1)
            v1, c1 = self.getSimilarity(sense1, sense2, grt_Sense)
            v2, c2 = self.getSimilarity(sense2, sense1, grt_Sense)
        if len(sense2) > len(sense1):
            grt_Sense = len(sense2)
            v1, c1 = self.getSimilarity(sense2, sense1, grt_Sense)
            v2, c2 = self.getSimilarity(sense1, sense2, grt_Sense)
        return np.array(v1),np.array(v2),c1,c2
        
    def main(self, sentence1, sentence2):
        sense1 = self.wordSenseDisambiguation(sentence1)
        sense2 = self.wordSenseDisambiguation(sentence2)        
        v1,v2,c1,c2 = self.shortestPathDistance(sense1,sense2)
        dot = np.dot(v1,v2)
        print("dot", dot) # getting the dot product
        tow = (c1+c2)/1.8
        final_similarity = dot/tow
        print("similarity",final_similarity)

def load_model(model_name: str) -> tuple[AutoModelForVision2Seq, AutoProcessor]:
    model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

def run_inference(model: AutoModelForVision2Seq, processor: AutoProcessor, image: Image, question: str) -> str:
    inputs = processor(images=image, text=question, return_tensors="pt").to("cuda", torch.float16)
    with torch.no_grad():
        outputs = model.generate(**inputs)
    return processor.decode(outputs, skip_special_tokens=True)[0].strip()

def loss(loss_model, answer, target):
    return loss_model.main(answer, target)

def run_benchmark(model, processor, loss_model, benchmark_file: str):
    total_loss = 0
    total_questions = 0
    with open(benchmark_file, "r") as f:
        data = json.load(f)
        for item in data:
            question = item["question"]
            target = item["target"]
            image = item["image"]
            try:
                image = Image.open(image)
            except:
                image = Image.open(image.astype("uint8") * 255)
            answer = run_inference(model, processor, image, question)
            total_loss += loss(loss_model, answer, target)
            total_questions += 1
    return total_loss / total_questions

def eval_model(model: AutoModelForVision2Seq, processor: AutoProcessor):
    loss_model = SentenceSimilarity()
    benchmarks = [
        "./benchmarks/A-VQA-20-1.json",
        "./benchmarks/A-VQA-20-2.json",
        "./benchmarks/A-VQA-20-3.json",
        "./benchmarks/A-VQA-20-4.json",
        "./benchmarks/A-VQA-20-5.json",
        "./benchmarks/MATHVQA-20.json",
        "./benchmarks/VQA-20.json",
        "./benchmarks/OKVQA-20.json"
    ]
    losses = {}
    for benchmark in tqdm(benchmarks):
        losses[benchmark] = run_benchmark(model, processor, loss_model, benchmark)
    return losses