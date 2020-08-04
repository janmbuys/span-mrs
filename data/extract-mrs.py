# partial source: https://github.com/goodmami/mrs-to-penman/mrs_to_penman.py

import sys
import os
import re
import argparse
import json

from delphin import tsql as d_tsql
from delphin import itsdb as d_itsdb
from delphin import dmrs as d_dmrs
from delphin import derivation as d_derivation
from delphin import tokens as d_tokens
from delphin.codecs import simplemrs as d_simplemrs
from delphin import mrs as d_mrs

def read_profile_old(dirname):
    ts = d_itsdb.TestSuite(dirname)
    if dirname[-1] == '/':
        dirname = dirname[:-1]
    profile_name = dirname[dirname.rindex('/')+1:]

    sentences = dict()
    untokenized_sentences = dict()
    tokenss = dict()
    parses = dict()
    total_parsed = 0
    total_not_parsed = 0

    for item in ts['item']:
        sentence_id = item['i-id']
        sentence = item['i-input']
        sentences[sentence_id] = sentence

    for parse in ts['parse']:
        if parse['p-tokens'] is not None:
            assert parse['i-id'] in sentences, "No matching sentence: " + str(parse[:])
            tokens_rep = d_tokens.YYTokenLattice.from_string(parse['p-tokens'])
            tokenss[parse['parse-id']] = (sentences[parse['i-id']], tokens_rep)
        elif parse['i-id'] in sentences:
            untokenized_sentences[parse['i-id']] = sentence
    
    for result in ts['result']:
        parse_id = result['parse-id']
        assert parse_id in tokenss, "Unmatched parse: " + parse_id
        constituency_tree = result['tree']
        derivation_tree = d_derivation.from_string(result['derivation'])

        try:
            mrs_rep = d_simplemrs.decode(result['mrs'])
            #mrs_rep = d_simplemrs.loads(result['mrs']) #TODO decode
        except d_mrs._exceptions.MRSSyntaxError:
            print("Skipping: MRS syntax error", result['mrs'])
            continue

        try: 
            #assert len(mrs_rep) == 1, "No unique MRS in example"
            #dmrs_rep = d_dmrs.from_mrs(mrs_rep[0])
            dmrs_rep = d_dmrs.from_mrs(mrs_rep)
            if parse_id in parses:
                print("multiple parses for same tokens")
            #assert parse_id not in parses, "multiple parses for same tokens"
            parses[parse_id] = (derivation_tree, dmrs_rep)
            total_parsed += 1
        except KeyError:
            print("DMRS conversion error") #, result['mrs'])
            print(result['mrs'])
            print(mrs_rep)
            break
            total_not_parsed += 1
            continue
    print("Parsed:", total_parsed, "Not parsed:", total_not_parsed)


def read_profile(dirname):
    ts = d_itsdb.TestSuite(dirname)
    if dirname[-1] == '/':
        dirname = dirname[:-1]
    profile_name = dirname[dirname.rindex('/')+1:]
    parses = []
  
    for sentence, parse_tokens, result_derivation, result_mrs in d_tsql.select('i-input p-tokens derivation mrs', ts):
        tokens_rep = d_tokens.YYTokenLattice.from_string(parse_tokens)
        derivation_tree = d_derivation.from_string(result_derivation)
        try:
            mrs_rep = d_simplemrs.decode(result_mrs)
        except d_mrs._exceptions.MRSSyntaxError:
            print("Skipping: MRS syntax error", result_mrs)
            continue
      
        dmrs_rep = d_dmrs.from_mrs(mrs_rep)
        parses.append((sentence, tokens_rep, mrs_rep, dmrs_rep))
    print("Parsed", len(parses))


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--input', help='directory path to a profile')
    args = argparser.parse_args()

    assert args.input and os.path.isdir(args.input), "Invalid input path"
    
    read_profile(args.input)

if __name__ == '__main__':
    main()

