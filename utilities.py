#!/usr/bin/python3
"""
Contains a smorgasbord of helpful methods meant to hide the complexity from the rest of the scripts.
"""

from glob import glob
from os.path import dirname, join, exists, isfile, basename
from nltk.tokenize.api import TokenizerI
from nltk.corpus import stopwords
import re
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec, Phrases
import json
english_stopwords = set(stopwords.words('english') + 
  ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "may", "june", "july", "aug", "sept", "oct", "nov", "dec",
    "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019",
    ] +
  ["am", "pm", "''", "--"]
  )
invariant_stopregexes = [re.compile(regex) for regex in [r"mr_", r"ms_", r"mrs_", r"dr_"]]


# if you'd like to exclude certain meaningless boilerplate from any particular corpus 
# include it in `boilerplate_by_class.json`
# examples might be site chrome if your corpus is text scraped from the web.
# otherwise, this meaningless boilerplate will dominate the "distinctive" words.
# e.g. {"class1": ["menu login about contact", "remember to follow me on twitter"], class2: ["facebook twitter linkedin", "contact us about corporate"]}
generic_boilerplates_fn = "boilerplate_by_class.json"
if exists(generic_boilerplates_fn):
  boilerplate_phrases = json.loads(open(generic_boilerplates_fn, 'r').read())
else:
  boilerplate_phrases = {}


def classes(input_folder, class_type="dirs"):
  return folders_as_classes(input_folder) if class_type == "dirs" else filenames_as_classes(input_folder)
def folders_as_classes(input_folder):
  """gets the classes that we're going to be finding distinctive words for, i.e. filenames in ARGV[1]"""
  return [basename(subdir) for subdir in glob(input_folder + ("*" if "*" not in input_folder else "")) if exists(subdir) and not isfile(subdir)]

def filenames_as_classes(input_folder):
  return [basename(filename) for filename in glob(input_folder + ("*" if "*" not in input_folder else "")) if exists(filename) and isfile(filename)]

def get_texts(class_name, input_folder, class_type="dirs"):
  """returns an iterable of tokenized sentences for the given class """
  # TODO: should return chained iterable of all the documents in the folder
  # so that a document is a file, not a line in a file.
  # should return a tag for that document too.
  return [ filename for filename in glob(join(input_folder.replace("*", ''), class_name, "*") if class_type == "dirs" else join(input_folder.replace("*", ''), class_name)) if exists(filename) and isfile(filename)]

def folder_name_to_model_name(input_folder):
  return join(dirname(__file__), "models", "doc2vec_model_{}.model".format(input_folder.replace("/", "_")))

most_recent_model_filename_filename = join(dirname(__file__), '.texttoolkit_model_filename')
def load_model(args_dot_model):
  if args_dot_model:
    model_name = args_dot_model
  else:
    if not exists(most_recent_model_filename_filename):
      print("Oops! You need to train a model before you can use this function (or specify the location of your model with the --model flag).")
      print("In most cases, running `python train_doc2vec_model.py inputs/your/input/folder` should do the trick!")
      exit(1)
    with open(most_recent_model_filename_filename, "r") as f:
        model_name =  (f.read().strip())
    if not exists(model_name):
      print("Oops! Your model may have been deleted or moved. Either specify its location with the --model flag or fix the path in .texttoolkit_model_filename (or train a new model).")
      exit(1)
  return Doc2Vec.load(model_name)
global bigrams_model
global trigrams_model
bigrams_model = None
trigrams_model = None
def load_ngrams_models():
  global bigrams_model
  global trigrams_model
  bigrams_model_name = join(dirname(__file__), "../texttoolkit/models/bigrams_phraser.bin")
  if exists(bigrams_model_name):
    if not bigrams_model:
      bigrams_model =  Phrases.load(bigrams_model_name)
  else:
    print("oops, couldn't find `models/bigrams_phraser.bin`. Try rerunning `$ bash setup.sh`")
    exit(1)


  trigrams_model_name = join(dirname(__file__), "../texttoolkit/models/trigrams_phraser.bin")
  if exists(trigrams_model_name):
    if not trigrams_model:
      trigrams_model = Phrases.load(trigrams_model_name)
  else:
    print("oops, couldn't find `models/trigrams_phraser.bin`. Try rerunning `$ bash setup.sh`")
    exit(1)

def to_tagged_document(file_body, ngrams_lambda, tags, document_class):
  text = ' '.join(file_body.split("\n"))
  boilerplatelets = boilerplate_phrases.get(document_class, [])
  boilerplateless_body = text.strip()
  for boilerplatelet in boilerplatelets:
    boilerplateless_body = boilerplateless_body.replace(boilerplatelet, '')
  tokens = boilerplateless_body.replace(".", "").split(" ")

  [token for token in tokens if token in english_stopwords and "%" not in token and not any(map(lambda re: re.search(token), invariant_stopregexes))]

  if not tokens or len(tokens) < 50:
    return None  # ignore short articles and various meta-articles
  return TaggedDocument( words=ngrams_lambda(tokens), 
                        tags=tags
                      )

class GenericIterator(object):
  cursor = None
  length = 0
  ngrams_models = {
    "bigrams": lambda x: bigrams_model[x],
    "trigrams": lambda x: trigrams_model[bigrams_model[x]],
  }



# an iterator, via http://rare-technologies.com/word2vec-tutorial/
class GenericDirectoryIterator(GenericIterator):
  def __init__(self, input_folder, document_class, ngrams_type=None, class_type="dirs"):
    self.input_folder = input_folder
    self.class_type = class_type
    self.document_class = document_class
    self.ngrams_type = ngrams_type
    if self.ngrams_type:
      load_ngrams_models()

  def __iter__(self):
    for document_filename in get_texts(self.document_class, self.input_folder, self.class_type):
      with open(document_filename, 'r') as file:
        self.length += 1
        tagged_document = to_tagged_document(
          file.read(),
          lambda tokens: self.ngrams_models[self.ngrams_type](tokens) if self.ngrams_type and self.ngrams_models.get(self.ngrams_type, False) else tokens, 
          ["filename-" + document_filename, "idx-" + self.document_class + str(self.length), "dc-" + self.document_class], 
          self.document_class
        )

        if not tagged_document:
          continue  # ignore short articles and various meta-articles
        yield tagged_document

class JsonlIterator(GenericIterator):
  """ for a JSONL file on  split_filings_into_paragraphs.py """
  def __init__(self, input_filename):
    self.input_filename = input_filename
    with open(self.input_filename, 'r') as f:
      self.length = len(f.read().split("\n"))
    if self.ngrams_type:
      load_ngrams_models()


  def __iter__(self):
    with open(self.input_filename):
      for line in open(self.input_filename, 'r').readlines():
        doc = json.loads(line)
        yield TaggedDocument(doc["_source"]["content"], doc["_source"]["path"])

class SugarcaneJsonlIterator(GenericIterator):
  """ for a JSONL file on  split_filings_into_paragraphs.py """
  def __init__(self, input_filename, ngrams_type=None):
    self.input_filename = input_filename
    self.ngrams_type=ngrams_type
    with open(self.input_filename, 'r') as f:
      self.length = len(f.read().split("\n"))
    if self.ngrams_type:
      load_ngrams_models()
    self.page_length = 1000


  def __iter__(self):
    with open(self.input_filename):
      for line in open(self.input_filename, 'r').readlines():
        doc = json.loads(line)
        tagged_document = to_tagged_document(doc["_source"]["content"], 
          lambda tokens: self.ngrams_models[self.ngrams_type](tokens) if self.ngrams_type and self.ngrams_models.get(self.ngrams_type, False) else tokens, 
          [
            # doc["_source"]["path"], 
            doc["_id"],
          ],
          ""
          )
        if not tagged_document:
          continue  # ignore short articles and various meta-articles
        yield tagged_document
