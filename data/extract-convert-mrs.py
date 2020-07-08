import sys
import os
import re
import argparse
import json

from delphin import itsdb as d_itsdb
from delphin import dmrs as d_dmrs
from delphin import derivation as d_derivation
from delphin import predicate as d_predicate
from delphin import tokens as d_tokens
from delphin.codecs import simplemrs as d_simplemrs
from delphin import mrs as d_mrs


class SemanticNode():
    def __init__(self, node_id, predicate):
        self.node_id = node_id
        self.predicate = predicate

        self.internal_child = -1
        self.internal_edge_label = ""
        self.is_semantic_terminal = False # no children, used for semantic roles
        self.is_semantic_receiver = False # has a parent up in the tree

    def __str__(self):
        return "%s" % (self.predicate)


class Node():
    def __init__(self, node_id, syntax_label, start_token_index, end_token_index):
        self.node_id = node_id
        self.syntax_labels = [syntax_label]
        self.phrase_labels = []
        self.semantic_nodes = []
        
        self.child_node_ids = []
        self.overlapping_node_ids = []

        self.start_token_index = start_token_index
        self.end_token_index = end_token_index # inclusive
        self.isToken = False
        self.token_form = ""
        self.token_ids = []

    def __str__(self):
        return "%d:%d %s %s children:%d tokens: %s" % (self.start_token_index, self.end_token_index, str(self.syntax_labels), str(self.semantic_nodes), len(self.child_node_ids), str(self.token_ids))


class Token():
    def __init__(self, token_id, tok=None):
        self.token_id = token_id
        if tok is not None:
            self.token_str = tok.form
            self.index = tok.start
            self.start_char = tok.lnk.data[0]
            self.end_char = tok.lnk.data[1]
        else:
            self.token_str = ""
            self.index = -1
            self.start_char = -1
            self.end_char = -1 # exclusive
        #self.unmatchedToken = False
        self.lemma = ""
        self.carg = ""
        self.isGrammarUnknown = False

    def __str__(self):
        return "%d %s" % (self.index, self.token_str)


def get_profile_name(dirname):
    if dirname[-1] == '/':
        dirname = dirname[:-1]
    return dirname[dirname.rindex('/')+1:]


def get_sentences(items):
    sentences = dict()
    for item in items:
        sentence_id = item['i-id']
        sentences[sentence_id] = item['i-input']
    return sentences


def get_tokenization(items, parses):
    sentences = get_sentences(items)

    tokenization_dict = dict()
    for parse in parses:
        if parse['p-tokens'] is not None:
            assert parse['i-id'] in sentences, "No matching sentence: " + str(parse[:])
            tokens_rep = d_tokens.YYTokenLattice.from_string(parse['p-tokens'])
            token_dict = {tok.id : tok for tok in tokens_rep.tokens}
            tokenization_dict[parse['parse-id']] = (sentences[parse['i-id']], token_dict)
    return tokenization_dict


def get_dmrs(mrs_str):
    # read and convert MRS
    try:
        mrs_rep = d_simplemrs.loads(mrs_str)
        assert len(mrs_rep) == 1, "No unique MRS in example"
        try: 
            dmrs_rep = d_dmrs.from_mrs(mrs_rep[0])
            return dmrs_rep
        except KeyError:
            #print("DMRS conversion error", result['mrs'])
            return None

    except d_mrs._exceptions.MRSSyntaxError:
        #print("Skipping: MRS syntax error", result['mrs'])
        return None


def parse_token_tfs(node_token):
    tfs = node_token.tfs
    start_char, end_char, form = -1, -1, ""
    #print(tfs)

    if "+FROM #1=\\" in tfs:
        start_i = tfs.index("+FROM #1=\\") + len("+FROM #1=\\") + 1
        start_char = int(tfs[start_i:tfs.index("\\", start_i)])
    elif "+FROM \\" in tfs:
        start_i = tfs.index("+FROM \\") + len("+FROM \\") + 1
        start_char = int(tfs[start_i:tfs.index("\\", start_i)])

    if "+TO \\" in tfs:
        end_i = tfs.index("+TO \\") + len("+TO \\") + 1
        end_char = int(tfs[end_i:tfs.index("\\", end_i)])

    if "+FORM \\" in tfs:
        form_i = tfs.index("+FORM \\") + len("+FORM \\") + 1
        form = tfs[form_i:tfs.index("\\", start_i)]

    return start_char, end_char, form


def match_token_vertex(new_token, token_dict):
    # match the start vertex of any token with same start char
    start_index, end_index = -1, -1

    for t_token in token_dict.values():
        if t_token.lnk.data[0] == new_token.start_char:
            start_index = t_token.start
            break

    for t_token in token_dict.values():
        if t_token.lnk.data[1] == new_token.end_char:
            end_index = t_token.end - 1
            break

    if start_index >= 0 and end_index == -1:
        end_index = start_index

    return start_index, end_index


def parse_node_token(node_token, token_dict):
    token_node_id = node_token.id

    if token_node_id in token_dict:
        tok = token_dict[token_node_id]
        if tok.start == tok.end-1:
            new_token = Token(token_node_id, tok)
            return [new_token]
        else:
            new_tokens = []
            # Split up multitokens
            for i in range(tok.start, tok.end):
                matched = False
                for tid, ttoken in token_dict.items():
                    if ttoken.start == i and ttoken.end == i+1:
                        new_tokens.append(Token(ttoken.id, ttoken)) 
                        matched = True
                        break
                assert matched, "Unmatched token in multitoken"
            return new_tokens
    else:
        new_token = Token(token_node_id)
        new_token.start_char, new_token.end_char, new_token.token_str = parse_token_tfs(node_token)
        assert new_token.start_char >= 0 and new_token.end_char >= 0, "Can't parse token " + str(tfs)

        new_token.index, end_index = match_token_vertex(new_token, token_dict)
        if new_token.index != end_index:
            print("Unmatched multitoken")
            print(node_token)
            print(new_token)
        assert new_token.index >= 0, "No matching token for derivation token " + str(tfs)
        return [new_token]


def create_span_node_map(nodes):
    span_node_map = dict()
    for node_id, node in nodes.items():
        tok_span = "%d:%d" % (node.start_token_index, node.end_token_index)
        span_node_map[tok_span] = node_id
    return span_node_map


def create_token_node_list(token_nodes):
    max_token_index = max([token.index for token in token_nodes.values()])
    assert len(token_nodes) == max_token_index +1, "Uncovered tokens in derivation"

    token_list = [-1 for _ in range(max_token_index+1)]
    for token_id, token_node in token_nodes.items():
        assert token_list[token_node.index] == -1, "Multiple derivation tokens with same index"
        token_list[token_node.index] = token_id

    return token_list


def create_char_token_maps(token_nodes):
    start_char_token_map = dict()
    end_char_token_map = dict()

    # Deals with mutiple tokens with same char span but different token spans
    for token_node_id, new_token in token_nodes.items():
        if new_token.start_char not in start_char_token_map:
            start_char_token_map[new_token.start_char] = token_node_id
        else:
            current_start = start_char_token_map[new_token.start_char]
            if token_nodes[current_start].index > new_token.index:
                start_char_token_map[new_token.start_char] = token_node_id

        if new_token.end_char not in end_char_token_map:
            end_char_token_map[new_token.end_char] = token_node_id
        else:
            current_end = end_char_token_map[new_token.end_char]
            if token_nodes[current_end].index < new_token.index:
                end_char_token_map[new_token.end_char] = token_node_id
    
    return start_char_token_map, end_char_token_map


def create_preterminal_node_map(nodes):
    token_preterminal_node_map = dict()
    for node in nodes.values():
        for token_id in node.token_ids:
            token_preterminal_node_map[token_id] = node.node_id

    return token_preterminal_node_map


class MeaningRepresentation():
    def __init__(self, sentence, token_dict, derivation_rep):
        self.sentence = sentence
        self.token_dict = token_dict
        self.nodes = dict()
        self.token_nodes = dict()

        assert len(derivation_rep.daughters) == 1 
        self.root_node_id = self.build_derivation_tree(derivation_rep.daughters[0])

        self.span_node_map = create_span_node_map(self.nodes)
        self.token_node_list = create_token_node_list(self.token_nodes)
        self.start_char_token_map, self.end_char_token_map = create_char_token_maps(self.token_nodes)
        self.token_preterminal_node_map = create_preterminal_node_map(self.nodes)

        self.dmrs_node_map = dict()
        self.token_sequence = [] # Ordered tokens. Format (token_id, token_position_index, form, lemma or unknown)
 

    def semantic_tree_str(self, node_id, level=0, newline=False, overlapping=False):
        node = self.nodes[node_id]
        out_str = "\n" + " "*level if newline else ""
        if overlapping:
            out_str += "*" + str(node.start_token_index) + ":" + str(node.end_token_index) + " "
        out_str += "(" + str([s_node.predicate for s_node in node.semantic_nodes]) + " "
        for child_id in node.overlapping_node_ids:
            out_str += self.semantic_tree_str(child_id, level+4, True, True) + " "
        for child_id in node.child_node_ids:
            join_line = len(node.overlapping_node_ids) == 0 and child_id == node.child_node_ids[0] and len(node.semantic_nodes) == 0
            out_str += self.semantic_tree_str(child_id, level+4, not join_line) + " "
        for token_id in node.token_ids:
            token_node = self.token_nodes[token_id]
            out_str += token_node.token_str + " "

        return out_str + ")"


    def build_derivation_tree(self, drv_node):
        new_node = Node(drv_node.id, drv_node.entity, drv_node.start, drv_node.end-1)

        # Collapse unary chains
        while type(drv_node) != d_derivation.UDFTerminal and len(drv_node.daughters) == 1:
            drv_node = drv_node.daughters[0]
            if type(drv_node) != d_derivation.UDFTerminal:
                assert drv_node.start == new_node.start_token_index and drv_node.end - 1 == new_node.end_token_index, "unary syntactic child has different span than parent"
                new_node.syntax_labels.append(drv_node.entity)
       
        if type(drv_node) == d_derivation.UDFTerminal:
            # Assign tokens to syntax nodes
            new_node.isToken = True
            new_node.token_form = drv_node.form
            for node_token in drv_node.tokens:
                new_tokens = parse_node_token(node_token, self.token_dict)
                for new_token in new_tokens:
                    new_node.token_ids.append(new_token.token_id)
                    self.token_nodes[new_token.token_id] = new_token
        else:
            # Recursively traverse children
            assert len(drv_node.daughters) == 2, "non-binary tree: " + sentence_id
            for child in drv_node.daughters:
                child_id = self.build_derivation_tree(child)
                new_node.child_node_ids.append(child_id)
            
        self.nodes[new_node.node_id] = new_node
        return new_node.node_id 


    def map_dmrs_node(self, node):
        assert node.lnk.data[0] in self.start_char_token_map and node.lnk.data[1] in self.end_char_token_map, "MRS predicate not matching tokens: " + str(node.predicate) + " " + str(node.lnk)
        start_node, end_node = self.start_char_token_map[node.lnk.data[0]], self.end_char_token_map[node.lnk.data[1]]
        start_token, end_token = self.token_nodes[start_node].index, self.token_nodes[end_node].index
        span_str = "%d:%d" % (start_token, end_token)
        matched_node = False

        # match nodes (not token nodes)
        if span_str in self.span_node_map:
            node_id = self.span_node_map[span_str]
            self.nodes[node_id].semantic_nodes.append(SemanticNode(node.id, node.predicate))
            self.dmrs_node_map[node.id] = (False, node_id)
            matched_node = True
        else:
            if start_token == end_token:
                print("DMRS Node matches to token, not node", node)
            self.dmrs_node_map[node.id] = (False, -1, start_token, end_token)
        
        return matched_node


    def match_overlapping_dmrs_node(self, node):
        # Find matching parent for unmatched node
        start_token, end_token = self.dmrs_node_map[node.id][2], self.dmrs_node_map[node.id][3]
        matched = False

        for node_id, s_node in self.nodes.items():
            if s_node.start_token_index == start_token and s_node.end_token_index == end_token:
                assert s_node.syntax_labels[0] == "NULL" # overlapping node
                self.nodes[node_id].semantic_nodes.append(SemanticNode(node.id, node.predicate))
                self.dmrs_node_map[node.id] = (False, node_id)
                matched = True
                break

        for node_id, s_node in self.nodes.items():
            if (not matched) and s_node.start_token_index <= start_token and s_node.end_token_index >= end_token and len(s_node.child_node_ids) == 2:
                left_node, right_node = self.nodes[s_node.child_node_ids[0]], self.nodes[s_node.child_node_ids[1]]   
                if left_node.end_token_index >= start_token and right_node.start_token_index <= end_token:
                    new_node_id = node.id
                    while new_node_id in self.nodes:
                        new_node_id = new_node_id*10

                    new_node = Node(new_node_id, "NULL", start_token, end_token)
                    new_node.semantic_nodes.append(SemanticNode(node.id, node.predicate))
                    self.nodes[node_id].overlapping_node_ids.append(new_node_id)
                    self.nodes[new_node_id] = new_node
                    self.dmrs_node_map[node.id] = (False, new_node_id)
                    matched = True
                    break

        assert matched, "Not Matched %s %d %d" % (str(node), start_token, end_token) 


    def link_dmrs_nodes(self, dmrs_rep):
        for edge in dmrs_rep.links:
            start_node_id = self.dmrs_node_map[edge.start][1]
            end_node_id = self.dmrs_node_map[edge.end][1]

            if start_node_id == end_node_id:
                # record internal edges
                node_id = start_node_id
                assert node_id in self.nodes
                node = self.nodes[node_id]
                sem_node_ids = [snode.node_id for snode in node.semantic_nodes]
                start_id, end_id = sem_node_ids.index(edge.start), sem_node_ids.index(edge.end)
                self.nodes[node_id].semantic_nodes[start_id].internal_child = sem_node_ids[end_id]
                self.nodes[node_id].semantic_nodes[start_id].internal_edge_label = edge.role
            #TODO figure this out
            #elif start_node_id in self.token_nodes:
            #    if end_node_id not in self.token_nodes:
            #        print(self.token_nodes[start_node_id].token_str, edge.role, [nd.predicate for nd in self.nodes[end_node_id].semantic_nodes])
            #else:
            #    if end_node_id in self.token_nodes:
            #        token_index = self.token_nodes[end_node_id].index
            #        if not (token_index >= self.nodes[start_node_id].start_token_index and token_index <= self.nodes[start_node_id].end_token_index):
            #            print([nd.predicate for nd in self.nodes[start_node_id].semantic_nodes], edge.role, self.token_nodes[end_node_id].token_str)
        #TODO order internal semantic nodes; find surface predicate


def read_profile(dirname):
    ts = d_itsdb.TestSuite(dirname)
    profile_name = get_profile_name(dirname)

    tokenization_dict = get_tokenization(ts['item'], ts['parse'])

    for result in ts['result']:
        #constituency_tree = result['tree'] #TODO put in derivation tree
        
        dmrs_rep = get_dmrs(result['mrs'])
        if dmrs_rep is None:
            continue

        assert result['parse-id'] in tokenization_dict, "Non tokenization for parse:" + str(result['parse-id'])
        sentence, token_dict = tokenization_dict[result['parse-id']]
        derivation_rep = d_derivation.from_string(result['derivation'])

        meaning_representation = MeaningRepresentation(sentence, token_dict, derivation_rep)

        # Map dmrs nodes to the meaning representation
        unmatched = False
        for node in dmrs_rep.nodes:
            matched_node = meaning_representation.map_dmrs_node(node)
            unmatched = unmatched or not matched_node

        # Map overlapping nodes
        for node in dmrs_rep.nodes:
            if meaning_representation.dmrs_node_map[node.id][1] == -1:
                meaning_representation.match_overlapping_dmrs_node(node)
                   
        meaning_representation.link_dmrs_nodes(dmrs_rep)

        #print(sentence)
        #print(meaning_representation.semantic_tree_str(meaning_representation.root_node_id))
        for node_id, node in meaning_representation.nodes.items():
            if len(node.overlapping_node_ids) > 0:
                 print(meaning_representation.semantic_tree_str(node_id))
            #if len(node.token_ids) > 1:
            #    print(meaning_representation.semantic_tree_str(node_id))
            has_surface = False
            for snode in node.semantic_nodes:
                if (not node.isToken) and d_predicate.is_surface(snode.predicate):
                    has_surface = True
            #if has_surface:
            #    print(meaning_representation.semantic_tree_str(node_id))
        #break

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--input', help='directory path to a profile')
    args = argparser.parse_args()
    assert args.input and os.path.isdir(args.input), "Invalid input path"
    
    read_profile(args.input)


if __name__ == '__main__':
    main()

