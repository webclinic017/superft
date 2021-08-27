import random


chatbot_dataset = """Me: How are you?
Friend: I am fine.
Me: What is your name?
Friend: My name is Bob.
Me: It is good to meet you, Bob.
Friend: It is nice to meet you too.
Me: Where do you live?
Friend: I live in Boston.
Me: What is your major?
Friend: I am studying computer science.
Me: What is your favorite food?
Friend: My favorite food is pizza.
Me: Me too!
Friend: How old are you?
Me: I am 20 years old.
Friend: Me too!
Me: Thanks for talking with me, Bob."""


class MarkovChainChatbot:
    def __init__(self, dataset):
        self.dataset = dataset
        self.tokens = self.tokenize()
        self.markov_chain = self.create_markov_chain()
        self.first_word = list(self.markov_chain.keys())[0]

    def tokenize(self):
        tokens = self.dataset.split('\n')
        tokens.pop(-1)
        return tokens

    def create_markov_chain(self):
        markov_chain = {}
        for i in range(len(self.tokens) - 1):
            if self.tokens[i] not in markov_chain:
                markov_chain[self.tokens[i]] = [self.tokens[i + 1]]
            else:
                markov_chain[self.tokens[i]].append(self.tokens[i + 1])
        return markov_chain

    def generate_sentence(self, length):
        current_word = self.first_word
        sentence = [current_word]
        for i in range(length):
            current_word = random.choice(self.markov_chain[current_word])
            sentence.append(current_word)
        return ' '.join(sentence)

chatbot = MarkovChainChatbot(chatbot_dataset)
print(chatbot.generate_sentence(10))