# This is a trial of statistical analysis. This is not configured to run.

# This uses the OpenIE library to extract the predicate phrases out of a single file

#import spacy
#from spacy.matcher import Matcher 
from openie import StanfordOpenIE
with StanfordOpenIE() as client:
	def get_triple(text):
		return client.annotate(text)
	    # for triple in client.annotate(text):
	    #     print('|-', triple)


	# def get_relation(sent):

	#   doc = nlp(sent)

	#   # Matcher class object 
	#   matcher = Matcher(nlp.vocab)

	#   #define the pattern 
	#   pattern = [{'DEP':'ROOT'}, 
	#             {'DEP':'prep','OP':"?"},
	#             {'DEP':'agent','OP':"?"},  
	#             {'POS':'ADJ','OP':"?"}] 

	#   matcher.add("matching_1", [pattern]) 

	#   matches = matcher(doc)
	#   k = len(matches) - 1

	#   span = doc[matches[k][1]:matches[k][2]] 

	#   return(span.text)

	# nlp = spacy.load('en_core_web_sm')

	def print_predicate(file, file2):

		lines = file.readlines()
		lines2 = file2.readlines()
		phrases = {}

		for line in lines:
			x = int(line)
			string = lines2[x-1]
			l = []
			l1 = get_triple(string)
			for dic in l1:
				if dic["relation"] not in l:
					l.append(dic["relation"])
			print("Line "+str(x)+" : "+str(l))

	file = open("0/sentences.txt", "r")

	file2 = open("0/1406.1078v3-Stanza-out.txt", "r")

	print_predicate(file, file2)

	
