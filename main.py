from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from indic_transliteration import sanscript
import re
import nltk
from nltk.corpus import words

nltk.download('words')

#english to hinglish
tokenizer = AutoTokenizer.from_pretrained("VasRosa/Hinglish-finetuned")
model = AutoModelForSeq2SeqLM.from_pretrained("VasRosa/Hinglish-finetuned")

def convert_hinglish_to_hindi(text):
    # Define a pattern to match Hinglish words (you may need to extend this pattern)
    hinglish_pattern = re.compile(r'[a-zA-Z]+')
    # Split the input text into words
    words = text.split()
    # Initialize the result list
    result = []
    for word in words:
        if hinglish_pattern.match(word):
            # Transliterate Hinglish words to Hindi using the indic-transliteration library
            hindi_word = sanscript.transliterate(word, sanscript.ITRANS, sanscript.DEVANAGARI)
            result.append(hindi_word)
        else:
            # Keep English words as they are
            result.append(word)
    # Join the words back into a sentence
    output_text = ' '.join(result)
    return output_text

english_words = set(words.words())

def is_english_word(word):
    return word.lower() in english_words

def compare_strings(string1, string2):
    words1 = string1.split()
    words2 = string2.split()
    output = []
    i = 0
    j = 0
    while i < len(words1) and j < len(words2):
        if is_english_word(words1[i]):
            output.append(words1[i])
            i += 1
            j += 1
        else:
            output.append(words2[j])
            i += 1
            j += 1
    return ' '.join(output)

while True:
    input_text = input("You: ")
    if input_text.lower() == "exit":
        break
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95, do_sample=True)
    hinglish_output = tokenizer.decode(output[0], skip_special_tokens=True)
    print("English Input: ",input_text)
    print("Hinglish Output: ",hinglish_output)
    output_text = convert_hinglish_to_hindi(hinglish_output)
    print("Hindi Output: ",output_text)
    output_string = compare_strings(hinglish_output, output_text)
    print("Hinglish Output: ",output_string,"\n")
