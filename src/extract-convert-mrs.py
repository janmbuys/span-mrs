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

import syntax
import semantics


def get_profile_name(dirname):
    if dirname[-1] == '/':
        dirname = dirname[:-1]
    return dirname[dirname.rindex('/')+1:]


def read_profile(args):
    ts = d_itsdb.TestSuite(args.input)
    profile_name = get_profile_name(args.input)
    lexicon = syntax.Lexicon(args.grammar)
 
    derivation_strs = []
    supertag_strs = []
    dmrs_json_strs = []

    for iid, sentence, parse_tokens, result_derivation, result_mrs in d_tsql.select('i-id i-input p-tokens derivation mrs', ts):
        tokens_rep = d_tokens.YYTokenLattice.from_string(parse_tokens)
        token_dict = {tok.id : tok for tok in tokens_rep.tokens}
        derivation_rep = d_derivation.from_string(result_derivation)

        try:
            mrs_rep = d_simplemrs.decode(result_mrs)
        except d_mrs._exceptions.MRSSyntaxError:
            #print("Skipping: MRS syntax error", result_mrs)
            continue

        dmrs_rep = d_dmrs.from_mrs(mrs_rep)

        mr = semantics.SemanticRepresentation(profile_name + ":" + iid, sentence, token_dict, derivation_rep, lexicon) # read derivation tree

        if args.convert_semantics:
            mr.map_dmrs(dmrs_rep)
            mr.process_semantic_tree(mr.root_node_id, dmrs_rep)

        mr.print_mrs()

        if args.extract_syntax:
            derivation_strs.append(mr.derivation_tree_str(mr.root_node_id, newline=False).lstrip())
            supertag_strs.append(mr.supertag_str(mr.root_node_id).strip())

        if args.extract_semantics:
            dmrs_json_strs.append(mr.dmrs_json_str(dmrs_rep))

    if args.extract_syntax:
        with open(args.output + ".dt", 'w') as dt_out:
            for s in derivation_strs:
                dt_out.write(s + "\n")
        with open(args.output + ".st", 'w') as st_out:
            for s in supertag_strs:
                st_out.write(s + "\n")

    if args.extract_semantics:
        with open(args.output + ".dmrs", 'w') as d_out:
            for s in dmrs_json_strs:
                if s != "":
                    d_out.write(s + "\n")
 

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--input', help='directory path to a profile')
    argparser.add_argument('-o', '--output', help='directory path to output')
    argparser.add_argument('-g', '--grammar', help='directory path to the ERG', default="data/original/erg1214")
    argparser.add_argument('--extract_syntax', action="store_true", help='extract derivation tree and supertags')
    argparser.add_argument('-c', '--convert_semantics', action="store_true", help='convert span-based DMRS')
    argparser.add_argument('--extract_semantics', action="store_true", help='convert span-based DMRS')

    args = argparser.parse_args()
    assert args.input and os.path.isdir(args.input), "Invalid input path"
    
    read_profile(args)


if __name__ == '__main__':
    main()

