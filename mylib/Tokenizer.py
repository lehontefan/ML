import regex

def word_tokenize(text, space=True):
    if space:
        return regex.findall(r" ?(?:\p{L}\.)+| ?\p{L}+|[.,!?;:()\"\\\'-]", text)
    else:
        return regex.findall(r"(?:\p{L}\.)+|\p{L}+|[.,!?;:()\"\\\'-]", text)

def byte_tokenize(text, space=True):
    words = word_tokenize(text, space)
    tokens = []
    for word in words:
        tokenized = [format(b, 'X') for b in word.encode('utf-8')]
        tokens.append(tokenized)
    return tokens

def bpe_tokenize(sequence, merges, space=True):
    new_sequence = byte_tokenize(sequence, space)
    for pair in merges:
        for index, word in enumerate(new_sequence):
            new_word = []
            i = 0
            while i < len(word):
                if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_sequence[index] = new_word
    return new_sequence