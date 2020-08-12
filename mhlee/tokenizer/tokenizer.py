from tokenizers import BertWordPieceTokenizer

# Initialize a tokenizer
tokenizer = BertWordPieceTokenizer()

# Then train it!
tokenizer.train([ "./sample.csv" ])

# Now, let's use it:
encoded = tokenizer.encode("미국에서는 여전히, 연준은 물론 정부와 의회 역시 신용경색 해소를 위해 다방면의 노력을 하고 있다. 하지만 그것은, 미 금융시스템의 붕괴는 모면케 해 줄 수 있을지언정, 순환적 경기침체까지 피해가게 만들 수는 없을 것 같다.")
print(encoded.ids)
print(encoded.tokens)
# And finally save it somewhere
tokenizer.save("./vocab.json")