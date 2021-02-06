
#Import Libraries----------------------------------------------------------

import tensorflow


from numpy import array
import string
from pickle import dump
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization
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

#load text
raw_text = load_doc('tiny-shakespeare.txt')

 
#clean// strip new lines to have one long sequence of characters
tokens = raw_text.split()
raw_text = ' '.join(tokens)

#Create Input-Output Seqs
length = 13
sequences = list()
for i in range(length, len(raw_text)):
	# select sequence of tokens
	seq = raw_text[i-length:i+1]
	# store
	sequences.append(seq)
print('Total Sequences:', len(sequences))

#Now we use the function save_doc to save the sequences
out_file = 'sequences.txt'
save_doc(sequences, out_file)

#TRAINING----------------------------------------------------------------------------------------------------

#load the sequence text
raw_text = load_doc('sequences.txt')


#Split the text by new lines. to be used below later
lines = raw_text.split('\n')


#Encoding seqs

#Sort the text, set of unique chars in text
chars = sorted(list(set(raw_text)))

#Possible characters
vocab_size = len(chars)
print('Total Possible Characters:', vocab_size)

#create a dictionary of char values to integer values
mapping = dict((c, i) for i, c in enumerate(chars))

#for each sequence in text, use the mapping above to find integer value for each char
sequences = list()
for line in lines:
	# integer encode line
	encoded_seq = [mapping[char] for char in line]
	# store
	sequences.append(encoded_seq)


#Aray Split into Input and Outputs
#create an array of the sequences
sequences = array(sequences)
#select all columns (except the last) as input
X = sequences[:,:-1]
#select last column as output
y = sequences[:,-1]

#One-Hot Encoding using Keras
#convert sequences to binary values
sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
#X is array of sequences
X = array(sequences)

#One-Hot encode the output
y = to_categorical(y, num_classes=vocab_size)

#Fit Model------------------------------------------------------------------------------------------------------------------------------------------------

#create layer-by layer model
model = Sequential()

model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2])))
model.add(BatchNormalization())
model.add(Dense(vocab_size, activation='softmax'))



# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


#Train for 50 epochs
model.fit(X, y, epochs=50, batch_size = 128, verbose=2)


# save the model to file
model.save('model.h5')

# save the mapping
dump(mapping, open('mapping.pkl', 'wb'))

#Text Generation------------------------------------------------------------------------

def generate_seq(model, mapping, seq_length, start_text, n_chars):

	# generate a fixed number of characters
	for _ in range(n_chars):
		
		encoded = [mapping[char] for char in start_text]
		
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# one hot encode
		encoded = to_categorical(encoded, num_classes=len(mapping))
		# predict character
		yhat = model.predict_classes(encoded, verbose=0)
		# reverse map integer to character
		out_char = ''
		for char, index in mapping.items():
			if index == yhat:
				out_char = char
				break
		# append to input
		start_text += char
	return start_text
 
# load the model
model = load_model('model.h5')

mapping = load(open('mapping.pkl', 'rb'))


print(generate_seq(model, mapping, 13, 'First Citizen', 500))









