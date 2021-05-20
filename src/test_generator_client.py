from __future__ import print_function

import random
import logging

import grpc

import test_generator_pb2
import test_generator_pb2_grpc

def get_sentences(stub):
    words = input("Words to generate cloze tests for: ")
    for word in words.split(' '):
        word_message = test_generator_pb2.Word(text=word)
        print("Searching for a sentence for '%s'..." %(word))
        sentence = stub.GetSentence(word_message)
        print(sentence.text)
    

def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = test_generator_pb2_grpc.TestGeneratorStub(channel)
        get_sentences(stub)
        


if __name__ == '__main__':
    logging.basicConfig()
    run()
