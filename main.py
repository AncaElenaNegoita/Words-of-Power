import requests
from time import sleep
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

orig_list = {
    1: {"text": "Feather", "cost": 1},
    2: {"text": "Coal", "cost": 1},
    3: {"text": "Pebble", "cost": 1},
    4: {"text": "Leaf", "cost": 2},
    5: {"text": "Paper", "cost": 2},
    6: {"text": "Rock", "cost": 2},
    7: {"text": "Water", "cost": 3},
    8: {"text": "Twig", "cost": 3},
    9: {"text": "Sword", "cost": 4},
    10: {"text": "Shield", "cost": 4},
    11: {"text": "Gun", "cost": 5},
    12: {"text": "Flame", "cost": 5},
    13: {"text": "Rope", "cost": 5},
    14: {"text": "Disease", "cost": 6},
    15: {"text": "Cure", "cost": 6},
    16: {"text": "Bacteria", "cost": 6},
    17: {"text": "Shadow", "cost": 7},
    18: {"text": "Light", "cost": 7},
    19: {"text": "Virus", "cost": 7},
    20: {"text": "Sound", "cost": 8},
    21: {"text": "Time", "cost": 8},
    22: {"text": "Fate", "cost": 8},
    23: {"text": "Earthquake", "cost": 9},
    24: {"text": "Storm", "cost": 9},
    25: {"text": "Vaccine", "cost": 9},
    26: {"text": "Logic", "cost": 10},
    27: {"text": "Gravity", "cost": 10},
    28: {"text": "Robots", "cost": 10},
    29: {"text": "Stone", "cost": 11},
    30: {"text": "Echo", "cost": 11},
    31: {"text": "Thunder", "cost": 12},
    32: {"text": "Karma", "cost": 12},
    33: {"text": "Wind", "cost": 13},
    34: {"text": "Ice", "cost": 13},
    35: {"text": "Sandstorm", "cost": 13},
    36: {"text": "Laser", "cost": 14},
    37: {"text": "Magma", "cost": 14},
    38: {"text": "Peace", "cost": 14},
    39: {"text": "Explosion", "cost": 15},
    40: {"text": "War", "cost": 15},
    41: {"text": "Enlightenment", "cost": 15},
    42: {"text": "Nuclear Bomb", "cost": 16},
    43: {"text": "Volcano", "cost": 16},
    44: {"text": "Whale", "cost": 17},
    45: {"text": "Earth", "cost": 17},
    46: {"text": "Moon", "cost": 17},
    47: {"text": "Star", "cost": 18},
    48: {"text": "Tsunami", "cost": 18},
    49: {"text": "Supernova", "cost": 19},
    50: {"text": "Antimatter", "cost": 19},
    51: {"text": "Plague", "cost": 20},
    52: {"text": "Rebirth", "cost": 20},
    53: {"text": "Tectonic Shift", "cost": 21},
    54: {"text": "Gamma-Ray Burst", "cost": 22},
    55: {"text": "Human Spirit", "cost": 23},
    56: {"text": "Apocalyptic Meteor", "cost": 24},
    57: {"text": "Earth’s Core", "cost": 25},
    58: {"text": "Neutron Star", "cost": 26},
    59: {"text": "Supermassive Black Hole", "cost": 35},
    60: {"text": "Entropy", "cost": 45},
}


our_words = [
    {"text": "Feather", "cost": 1, "counter": "Rock"},
    {"text": "Coal", "cost": 1, "counter": "Water"},
    {"text": "Pebble", "cost": 1, "counter": "Rock"},
    {"text": "Leaf", "cost": 2, "counter": "Flame"},
    {"text": "Paper", "cost": 2, "counter": "Flame"},
    {"text": "Rock", "cost": 2, "counter": "Water"},
    {"text": "Water", "cost": 3, "counter": "Disease"},
    {"text": "Twig", "cost": 3, "counter": "Flame"},
    {"text": "Sword", "cost": 4, "counter": "Shield"},
    {"text": "Shield", "cost": 4, "counter": "Gun"},
    {"text": "Gun", "cost": 5, "counter": "Laser"},
    {"text": "Flame", "cost": 5, "counter": "Water"},
    {"text": "Rope", "cost": 5, "counter": "Flame"},
    {"text": "Disease", "cost": 6, "counter": "Cure"},
    {"text": "Cure", "cost": 6, "counter": "Fate"},
    {"text": "Bacteria", "cost": 6, "counter": "Vaccine"},
    {"text": "Shadow", "cost": 7, "counter": "Light"},
    {"text": "Light", "cost": 7, "counter": "Storm"},
    {"text": "Virus", "cost": 7, "counter": "Vaccine"},
    {"text": "Sound", "cost": 8, "counter": "Time"},
    {"text": "Time", "cost": 8, "counter": "Fate"},
    {"text": "Fate", "cost": 8, "counter": "Rebirth"},
    {"text": "Earthquake", "cost": 9, "counter": "Earth"},
    {"text": "Storm", "cost": 9, "counter": "Earth"},
    {"text": "Vaccine", "cost": 9, "counter": "Fate"},
    {"text": "Logic", "cost": 10, "counter": "Fate"},
    {"text": "Gravity", "cost": 10, "counter": "Star"},
    {"text": "Robots", "cost": 10, "counter": "Virus"},
    {"text": "Stone", "cost": 11, "counter": "Water"},
    {"text": "Echo", "cost": 11, "counter": "Sound"},
    {"text": "Thunder", "cost": 12, "counter": "Earth"},
    {"text": "Karma", "cost": 12, "counter": "Fate"},
    {"text": "Wind", "cost": 13, "counter": "Explosion"},
    {"text": "Ice", "cost": 13, "counter": "Flame"},
    {"text": "Sandstorm", "cost": 13, "counter": "Explosion"},
    {"text": "Laser", "cost": 14, "counter": "Shield"},
    {"text": "Magma", "cost": 14, "counter": "Water"},
    {"text": "Peace", "cost": 14, "counter": "War"},
    {"text": "Explosion", "cost": 15, "counter": "Water"},
    {"text": "War", "cost": 15, "counter": "Nuclear Bomb"},
    {"text": "Enlightenment", "cost": 15, "counter": "Fate"},
    {"text": "Nuclear Bomb", "cost": 16, "counter": "Earth"},
    {"text": "Volcano", "cost": 16, "counter": "Water"},
    {"text": "Whale", "cost": 17, "counter": "Gun"},
    {"text": "Earth", "cost": 17, "counter": "Star"},
    {"text": "Moon", "cost": 17, "counter": "Gravity"},
    {"text": "Star", "cost": 18, "counter": "Supernova"},
    {"text": "Tsunami", "cost": 18, "counter": "Earth"},
    {"text": "Supernova", "cost": 19, "counter": "Rebirth"},
    {"text": "Antimatter", "cost": 19, "counter": "Earth"},
    {"text": "Plague", "cost": 20, "counter": "Vaccine"},
    {"text": "Rebirth", "cost": 20, "counter": "Karma"},
    {"text": "Tectonic Shift", "cost": 21, "counter": "Earth"},
    {"text": "Gamma-Ray Burst", "cost": 22, "counter": ""},
    {"text": "Human Spirit", "cost": 23, "counter": "Fate"},
    {"text": "Apocalyptic Meteor", "cost": 24, "counter": "Earth"},
    {"text": "Earth's Core", "cost": 25, "counter": "Magma"},
    {"text": "Neutron Star", "cost": 26, "counter": "Feather"},
    {"text": "Supermassive Black Hole", "cost": 35, "counter": "Feather"},
    {"text": "Entropy", "cost": 45, "counter": "Time"},


    {"text": "Building", "cost": 100, "counter": "Earthquake"},
    {"text": "Animal", "cost": 100, "counter": "Gun"},
    {"text": "Person", "cost": 100, "counter": "Gun"},
    {"text": "Flower", "cost": 100, "counter": "Flame"}, 
    {"text": "Electronic", "cost": 100, "counter": "Water"},
    {"text": "Fruit", "cost": 100, "counter": "Sword"}, 
    {"text": "Vegetables", "cost": 100, "counter": "Sword"}, 
    {"text": "Clothes", "cost": 100, "counter": "Flame"},
    {"text": "Object", "cost": 100, "counter": "Flame"},
    {"text": "Weather", "cost": 100, "counter": "Earth"},
    {"text": "Mithology", "cost": 100, "counter": "Karma"}, 
    {"text": "Celestial", "cost": 100, "counter": "Gravity"}, 
    {"text": "Educational Subject", "cost": 100, "counter": "Logic"},
    {"text": "Underworld", "cost": 100, "counter": "Flame"},
    {"text": "Magic", "cost": 100, "counter": "Logic"},
    {"text": "Death", "cost": 100, "counter": "Flame"},
    {"text": "Garden", "cost": 100, "counter": "Flame"},
    {"text": "Ustensils", "cost": 100, "counter": "Rock"},
    {"text": "Clean", "cost": 100, "counter": "Flame"},
    {"text": "Cup", "cost": 100, "counter": "Rock"},
    {"text": "Small Particles", "cost": 100, "counter": "Wind"},
    {"text": "Hand", "cost": 100, "counter": "Sword"},
    {"text": "Mess", "cost": 100, "counter": "Wind"},

]
valid_words = {word["text"] for word in our_words}
for word in our_words:
    if word["counter"] not in valid_words:
        potential_counters = [w for w in our_words if w["cost"] > word["cost"]]
        if potential_counters:
            word["counter"] = min(potential_counters, key=lambda x: x["cost"])["text"]
        else:
            word["counter"] = max(our_words, key=lambda x: x["cost"])["text"]

model = SentenceTransformer('./models/all-MiniLM-L6-v2/')

word_embeddings = {word["text"].lower(): model.encode(word["text"]) for word in our_words}

word_lookup = {word["text"].lower(): word for word in our_words}

class WordGameStrategy:
    def __init__(self):
        self.used_words = set()
        self.total_cost = 0
        self.round = 1
    
    def find_most_similar(self, opponent_word):
        opponent_embedding = model.encode(opponent_word.lower())
        similarities = []
        
        for word_text, embedding in word_embeddings.items():
            similarity = cosine_similarity([opponent_embedding], [embedding])[0][0]
            similarities.append((word_text, similarity))
        
        return max(similarities, key=lambda x: x[1])[0]
    
    def select_word(self, opponent_word):
        similar_word = self.find_most_similar(opponent_word)
        similar_word_data = word_lookup[similar_word.lower()]
        
        counter_word = similar_word_data["counter"]
        counter_data = word_lookup[counter_word.lower()]
        self.total_cost += counter_data["cost"]
        
        self.round += 1
        return counter_data

host = "http://172.18.4.158:8000"
post_url = f"{host}/submit-word"
get_url = f"{host}/get-word"
status_url = f"{host}/status"

NUM_ROUNDS = 5

def what_beats(word):
    counter_word = strategy.select_word(word)
    return next(key for key, value in orig_list.items() if value["text"] == counter_word["text"])


def play_game(player_id):

    for round_id in range(1, NUM_ROUNDS+1):
        round_num = -1
        while round_num != round_id:
            response = requests.get(get_url)
            print(response.json())
            sys_word = response.json()['word']
            round_num = response.json()['round']

        if round_id >= 1:
            status = requests.get(status_url)
            print(status.json())

        choosen_word = what_beats(sys_word)
        print(f"Chosen word: {orig_list[choosen_word]['text']}")
        data = {"player_id": player_id, "word_id": choosen_word, "round_id": round_id}
        response = requests.post(post_url, json=data)
        print(response.json())


strategy = WordGameStrategy()
play_game("q4K88XrnXk")
