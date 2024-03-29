import sys
import os
import argparse

from delphin import tsql as d_tsql
from delphin import itsdb as d_itsdb
from delphin import dmrs as d_dmrs
from delphin import derivation as d_derivation
from delphin import predicate as d_predicate
from delphin import tokens as d_tokens
from delphin.codecs import simplemrs as d_simplemrs
from delphin import mrs as d_mrs
from delphin import predicate as d_predicate

from delphin import ace as d_ace
#from delphin import repp as d_repp

import syntax
import semantics


def read_result(sentence, parse_tokens, result, lexicon, args):
    #for iid, sentence, parse_tokens, result_derivation, result_mrs in d_tsql.select('i-id i-input p-tokens derivation mrs', ts):
    tokens_rep = d_tokens.YYTokenLattice.from_string(parse_tokens)
    token_dict = {tok.id : tok for tok in tokens_rep.tokens}
    derivation_rep = d_derivation.from_string(result['derivation'])

    try:
        mrs_rep = d_simplemrs.decode(result['mrs']) 
    except d_mrs._exceptions.MRSSyntaxError:
        print("Skipping: MRS syntax error", result['mrs'])
        return

    dmrs_rep = d_dmrs.from_mrs(mrs_rep)

    mr = semantics.SemanticRepresentation("input:0", sentence, token_dict, derivation_rep, lexicon) # read derivation tree

    if args.convert_semantics:
        mr.map_dmrs(dmrs_rep)
        mr.process_semantic_tree(mr.root_node_id, dmrs_rep)

    if args.extract_syntax:
        print(mr.supertag_str(mr.root_node_id).strip())
        print(mr.derivation_tree_str(mr.root_node_id, newline=False).lstrip())

    if args.extract_semantics:
        print(mr.dmrs_json_str(dmrs_rep))


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-g', '--grammar', help='directory path to the ERG', default="data/original/erg1214")
    argparser.add_argument('--extract_syntax', action="store_true", help='extract derivation tree and supertags')
    argparser.add_argument('-c', '--convert_semantics', action="store_true", help='convert span-based DMRS')
    argparser.add_argument('--extract_semantics', action="store_true", help='convert span-based DMRS')

    args = argparser.parse_args()
    lexicon = syntax.Lexicon(args.grammar)

    sentence = input()
    with d_ace.ACEParser(args.grammar + '/erg-1214-x86-64-0.9.31.dat') as parser:
        response = parser.interact(sentence)
        #if "result" in response:
        result = response.result(0)
        #repp_tokenizer = d_repp.REPP.from_config(args.grammar + '/pet/repp.set')
        #tokens_rep = repp_tokenizer.tokenize(sentence).tokens
        tokens_rep = response['tokens']['initial']

        read_result(sentence, tokens_rep, result, lexicon, args)
        #else:
        #    print("No result")


if __name__ == '__main__':
    main()

