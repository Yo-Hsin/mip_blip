import nltk
from argparse import ArgumentParser
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk import pos_tag


nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

#caption = "A crowd of people standing in front of a street, one holding a sign and looking to her left." # people standing
#caption = "The large crowd of people is gathering for the people to arrive" # large crowd
#caption = "A girl sits on the bed and is afraid to hear something from her mother" # girl sits


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        '--caption', type=str, 
        default='A girl sits on the bed and is afraid to hear something from her mother') # girl sits
    return parser.parse_args()


def find_most_intense_phrase(caption):
    sia = SentimentIntensityAnalyzer()
    tokens = word_tokenize(caption)
    tagged = pos_tag(tokens)

    # Identify potential emotional phrases and calculate their sentiment intensity
    phrases = {}
    for i in range(len(tagged) - 1):
        if tagged[i][1].startswith('JJ') and tagged[i+1][1].startswith('NN'):  # Adjective + Noun
            phrase = tagged[i][0] + ' ' + tagged[i+1][0]
            intensity = sia.polarity_scores(phrase)['compound']
            phrases[phrase] = intensity
        elif tagged[i][1].startswith('NN') and tagged[i+1][1].startswith('VB'):  # Noun + Verb
            phrase = tagged[i][0] + ' ' + tagged[i+1][0]
            intensity = sia.polarity_scores(phrase)['compound']
            phrases[phrase] = intensity
        elif tagged[i][1].startswith('VB') and tagged[i+1][1].startswith('RB'):  # Verb + Adverb
            phrase = tagged[i][0] + ' ' + tagged[i+1][0]
            intensity = sia.polarity_scores(phrase)['compound']
            phrases[phrase] = intensity
        elif tagged[i][1].startswith('NN') and tagged[i+1][1].startswith('JJ'):  # Noun + Adjective
            phrase = tagged[i][0] + ' ' + tagged[i+1][0]
            intensity = sia.polarity_scores(phrase)['compound']
            phrases[phrase] = intensity
        elif tagged[i][1].startswith('JJ') and tagged[i+1][1].startswith('JJ'):  # Consecutive Adjectives
            phrase = tagged[i][0] + ' ' + tagged[i+1][0]
            intensity = sia.polarity_scores(phrase)['compound']
            phrases[phrase] = intensity

    # Select the most intense phrase
    max_intensity = -float('inf')
    most_intense_phrase = ""
    for phrase in phrases:
        intensity = sia.polarity_scores(phrase)['compound']
        if intensity > max_intensity:
            max_intensity = intensity
            most_intense_phrase = phrase

    return most_intense_phrase if most_intense_phrase else "No strong emotional expression found."


if __name__ == '__main__':
    args = parse_arguments()
    print(find_most_intense_phrase(args.caption))