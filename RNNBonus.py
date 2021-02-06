
#Import Libraries----------------------------------------------------------

import tensorflow
import string

from numpy import array
import string
from pickle import dump
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.layers import Embedding
from pickle import load

#------------------------------------------------------------------------------

#LOADING AND PRE-PROCESSING TEXT-----------------------------------------------------------------------

#function to load document
def load_doc(filename):
    file = open(filename, "r")
    text = file.read()
    file.close()
    return text


#function to save the sequences into a text file
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()


def clean_doc(doc):
	# replace '--' with a space ' '
	doc = doc.replace('--', ' ')
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# make lower case
	tokens = [word.lower() for word in tokens]
	return tokens


#load text
raw_text = load_doc('tiny-shakespeare.txt')

clean_text = clean_doc(raw_text)


#Create Input-Output Seqs
length = 5
sequences = list()
for i in range(length, len(clean_text)):
	# select sequence of tokens
	seq = clean_text[i-length:i+1]
	#convert to line
	line = ' '.join(seq)
	# store
	sequences.append(line)
print('Total Sequences:', len(sequences))

#Now we use the function save_doc to save the sequences
out_file = 'sequences.txt'
save_doc(sequences, out_file)

#TRAINING----------------------------------------------------------------------------------------------------

#load the sequence text
seq_text = load_doc('sequences.txt')


#Split the text by new lines. to be used below later
lines = seq_text.split('\n')


#Encoding seqs

#use Tokenizer to convert lists of words into list of integers
mapping = Tokenizer()
mapping.fit_on_texts(lines)
sequences = mapping.texts_to_sequences(lines)

vocab_size = len(mapping.word_index) + 1


#Possible characters
print('Total Possible Words:', vocab_size)

#Aray Split into Input and Outputs
#create an array of the sequences
sequences = array(sequences)
#select all columns (except the last) as input
X = sequences[:,:-1]
#select last column as output
y = sequences[:,-1]

#One-Hot encode the output
y = to_categorical(y, num_classes=vocab_size)

#Fit Model------------------------------------------------------------------------------------------------------------------------------------------------
#create layer-by layer model
model = Sequential()

model.add(Embedding(vocab_size, 50, input_length=X.shape[1]))


model.add(LSTM(256, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))



# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


#Train for 50 epochs
model.fit(X, y, epochs=120, batch_size = 128, verbose=2)


# save the model to file
model.save('model.h5')

# save the mapping
dump(mapping, open('mapping.pkl', 'wb'))

#Text Generation------------------------------------------------------------------------

def generate_seq(model, mapping, seq_length, start_text, n_chars):

	result = list()
	# generate a fixed number of characters
	for _ in range(n_chars):
		
		encoded = mapping.texts_to_sequences([start_text])[0]
		
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# predict
		yhat = model.predict_classes(encoded, verbose=0)
		# reverse map integer to character
		out_char = ''
		for char, index in mapping.word_index.items():
			if index == yhat:
				out_char = char
				break
		# append to input
		start_text += ' ' + out_char
		result.append(out_char)
	return ' '.join(result)
 
# load the model
model = load_model('model.h5')

mapping = load(open('mapping.pkl', 'rb'))


print(generate_seq(model, mapping, 5, 'First', 500))


















