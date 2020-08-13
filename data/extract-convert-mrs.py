import sys
import os
import re
import argparse
import json
import difflib

from delphin import tsql as d_tsql
from delphin import itsdb as d_itsdb
from delphin import dmrs as d_dmrs
from delphin import derivation as d_derivation
from delphin import predicate as d_predicate
from delphin import tokens as d_tokens
from delphin.codecs import simplemrs as d_simplemrs
from delphin import mrs as d_mrs
from delphin import tdl as d_tdl
from delphin import predicate as d_predicate


class Lexicon():
    def __init__(self, erg_path):
        self.lex = {} # map syntactic name (in tree) to lexical entry
        self.read_lexicon(erg_path)

    def read_lexicon(self, erg_path):
        #TODO extend to do more complex things
        for event, obj, _ in d_tdl.iterparse(erg_path + "/lexicon.tdl"):
            if event == 'TypeDefinition':
                syntactic_name = obj.identifier
                lexical_type = str(obj.conjunction.terms[0])
                lemma = " ".join(map(str, obj['ORTH'].values())) 
                if "SYNSEM" in obj: # else doesn't really matter
                    semantic_predicate, carg = None, None
                    if "LKEYS.KEYREL.PRED" in obj["SYNSEM"]:
                        semantic_predicate = str(obj["SYNSEM"]["LKEYS.KEYREL.PRED"])
                    if "LKEYS.KEYREL.CARG" in obj["SYNSEM"]:
                        carg = str(obj["SYNSEM"]["LKEYS.KEYREL.CARG"])
                    self.lex[syntactic_name] = (lemma, lexical_type, semantic_predicate, carg)

    def get_lexical_type(self, name):
        if name in self.lex:
            return self.lex[name][1]
        else:
            return None


class SemanticNode():
    def __init__(self, node_id, predicate, carg):
        self.node_id = node_id
        self.original_predicate = predicate
        self.predicate = predicate
        self.carg = None if carg == "" else carg

        self.internal_parent = -1
        self.internal_child = -1
        self.internal_edge_label = ""
        self.has_ancestor = False
        self.is_semantic_terminal = False # no children, used for semantic roles
        self.is_semantic_head = False # has direct semantic children in tree
        self.is_surface = False

    def __str__(self):
        return "%s" % (self.original_predicate)
        #return "%s %s %s" % (self.original_predicate, self.carg, self.predicate)


class Node():
    def __init__(self, node_id, syntax_label, start_token_index, end_token_index):
        self.node_id = node_id
        self.syntax_labels = [syntax_label]
        self.phrase_labels = []
        self.semantic_nodes = [] #TODO need to store indices here to make life easier

        self.semantic_parent_node = -1
        self.semantic_parent_edge_label = ""
        
        self.child_node_ids = []
        self.overlapping_node_ids = []

        self.start_token_index = start_token_index
        self.end_token_index = end_token_index # inclusive
        self.isToken = False
        self.token_form = ""
        self.token_ids = []

    def __str__(self):
        return str([str(snode) for snode in self.semantic_nodes])
        #return "%d:%d %s %s children:%d tokens: %s" % (self.start_token_index, self.end_token_index, str(self.syntax_labels), str(self.semantic_nodes), len(self.child_node_ids), str(self.token_ids))


class Token():
    def __init__(self, token_id, tok=None):
        self.token_id = token_id
        if tok is not None:
            self.token_str = tok.form # TODO want the proper string from sentence here
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
        self.is_unknown = False

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
        except KeyError: # "DMRS conversion error"
            return None

    except d_mrs._exceptions.MRSSyntaxError:
        return None


def parse_token_tfs(node_token):
    tfs = node_token.tfs
    start_char, end_char, form = -1, -1, ""

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
        form_i = tfs.index("+FORM \\") + len("+FORM \\") + 1 #TODO bug in finding end
        form = tfs[form_i:tfs.index("\\", start_i)]
        #print(form)

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

        new_token.index, _ = match_token_vertex(new_token, token_dict)
        assert new_token.index >= 0, "No matching token for derivation token " + str(tfs)
        return [new_token]


def covers_span(parent_node, child_node):
    return parent_node.start_token_index <= child_node.start_token_index and parent_node.end_token_index >= child_node.end_token_index


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


def text_to_digits(t):
    digit_map = {"zero":0, "one":1, "two":2, "three":3, "four":4, "five":5, "six":6, "seven":7, "eight":8, "nine":9, "ten":10, "eleven":11, "twelve":12, "thirteen":13, "fourteen":14, "fifteen":15, "sixteen":16, "seventeen":17, "eighteen":18, "nineteen":19, "twenty":20, "thirty":30, "forty":40, "fifty":50, "sixty":60, "seventy":70, "eighty":80, "ninety":90, "hundred":100, "thousand":1000, "million":1000000, "billion":1000000000, "trillion":1000000000000, "second":2, "third":3, "fifth":5, "eighth":8, "ninth":9, "half":"1/2"}
    if t in digit_map:
        return str(digit_map[t])
    elif t + "th" in digit_map:
        return str(digit_map[t+"th"])
    else:
        return t


def clean_token_lemma(tok_str, pred_is_digit=False):
    t_str = tok_str.lower().strip("-.,\"\';")
    # Manual lemmatization rules
    surface_map = {"best":"good", "better":"good", "/":"and", "is":"be", "was":"be", "km":"kilometer", "worst":"bad", "worse":"bad", "&":"and", "’s":"be", "'s":"be", "isn’t":"be", "wasn’t":"be", "%":"percent", "$":"dollar", "went":"go", ":":"colon", "are":"be", "were":"be"}
    if t_str in surface_map:
        t_str = surface_map[t_str]
    if pred_is_digit:
        t_str = text_to_digits(t_str)
    return t_str


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


def match_surface_predicate_token(predicate, start_token_index, end_token_index, token_nodes, token_node_list):
    match_prob = []
    tkns = []
    lemma, pos, sense = d_predicate.split(predicate)
    if pos == 'u':
        lemma = lemma[:lemma.rindex('/')]
    for tok_index in range(start_token_index, end_token_index+1):
        # id lookup bypasses the derivation node
        token = token_nodes[token_node_list[tok_index]] 
        t_str = clean_token_lemma(token.token_str, predicate.isdigit())
        tkns.append(t_str)
        seq = difflib.SequenceMatcher(a=t_str, b=lemma.lower())
        match_prob.append(seq.ratio())
    token_match = start_token_index + match_prob.index(max(match_prob))
    if max(match_prob) == 0:
        print(predicate, tkns)

    return token_match


class SyntacticRepresentation():
    def __init__(self, sid, sentence, token_dict, derivation_rep, lexicon=None):
        self.sid = sid
        self.sentence = sentence
        self.token_dict = token_dict
        self.nodes = dict()
        self.token_nodes = dict()
        self.lexicon = lexicon

        assert len(derivation_rep.daughters) == 1 
        self.root_node_id = self.build_derivation_tree(derivation_rep.daughters[0])

        self.span_node_map = create_span_node_map(self.nodes)
        self.token_node_list = create_token_node_list(self.token_nodes)
        self.start_char_token_map, self.end_char_token_map = create_char_token_maps(self.token_nodes)
        self.token_preterminal_node_map = create_preterminal_node_map(self.nodes)

        self.token_sequence = [] # Ordered tokens. Format (token_id, token_position_index, form, lemma or unknown) #TODO 


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

    def derivation_tree_str(self, node_id, level=0, newline=False):
        node = self.nodes[node_id]
        out_str = "\n" + " "*level if newline else ""

        unary_count = 0 
        for label in node.syntax_labels:
            if label.endswith("_c") and not (len(node.token_ids) > 0 and label == node.syntax_labels[-1]):
                out_str += "(" + label + " "
                unary_count += 1
            else:
                break

        out_str += " ".join([self.derivation_tree_str(child_id, level+4, newline) for child_id in node.child_node_ids])

        out_str += " ".join(['(X ' + self.token_nodes[token_id].token_str + ')' for token_id in node.token_ids])

        out_str += ")"*unary_count
        return out_str


    def supertag_str(self, node_id):
        tags, words = self.supertag_tuple(node_id)
        tag_str = ""
        for tag, word in zip(tags, words):
            tag_str += "[" + ";".join(tag) + "] " + " ".join(word) + " "
        return tag_str.strip()


    def supertag_tuple(self, node_id):
        node = self.nodes[node_id]

        tag = []
        tag_state = False
        for label in node.syntax_labels:
            if tag_state or (not label.endswith("_c")) or (len(node.token_ids) > 0 and label == node.syntax_labels[-1]):
                tag_state = True
                tag.append(label)

        if len(tag) > 0:
            lex_type = self.lexicon.get_lexical_type(tag[-1])
            if lex_type is not None:
                tag[-1] = lex_type
            elif not (tag[-1].startswith("generic") or tag[-1].startswith("punct_")):
                print(tag[-1])

        assert not (len(tag) > 0 and len(node.child_node_ids) > 0), tag

        tags = []
        words = []

        for child_id in node.child_node_ids:
            child_tags, child_words = self.supertag_tuple(child_id)
            tags.extend(child_tags)
            words.extend(child_words)

        word = [self.token_nodes[token_id].token_str for token_id in node.token_ids]
        if len(word) > 0:
            assert len(tag) > 0, word
            tags = [tag]
            words = [word]

        return tags, words


class SemanticRepresentation(SyntacticRepresentation):
    def __init__(self, sid, sentence, token_dict, derivation_rep, lexicon=None):
        super().__init__(sid, sentence, token_dict, derivation_rep, lexicon)

        self.dmrs_node_map = dict()
        self.overlapping_node_map = dict()
 

    def semantic_tree_str(self, node_id, level=0, newline=False, overlapping=False):
        node = self.nodes[node_id]
        out_str = "\n" + " "*level if newline else ""
        if overlapping:
            out_str += "*" + str(node.start_token_index) + ":" + str(node.end_token_index) + " "
        out_str += "(" + str([str(s_node.node_id) + ":" + s_node.predicate for s_node in node.semantic_nodes]) + " "
        for child_id in node.overlapping_node_ids:
            out_str += self.semantic_tree_str(child_id, level+4, True, True) + " "
        for child_id in node.child_node_ids:
            join_line = len(node.overlapping_node_ids) == 0 and child_id == node.child_node_ids[0] and len(node.semantic_nodes) == 0
            out_str += self.semantic_tree_str(child_id, level+4, not join_line) + " "
        for token_id in node.token_ids:
            token_node = self.token_nodes[token_id]
            if token_node.is_unknown:
                out_str += "{<unk>} "
            elif token_node.lemma == "":
                out_str += "{<none>} "
            else:
                out_str += "{%s} " % (token_node.lemma)
            out_str += token_node.token_str + " "

        return out_str + ")"
    
    
    def dmrs_json_str(self, dmrs_rep):
        dmrs_dict = {}
        dmrs_dict["id"] = self.sid.split(":")[1]
        dmrs_dict["source"] = self.sid.split(":")[0]
        dmrs_dict["input"] = self.sentence
        dmrs_dict["tokens"] = []
        dmrs_dict["nodes"] = []
        dmrs_dict["edges"] = []

        for i, token_id in enumerate(self.token_node_list):
            token = self.token_nodes[token_id]
            dmrs_dict["tokens"].append({"index": i, "form": token.token_str, "lemma": token.token_str if token.lemma == "" else token.lemma})
            if token.carg != "":
                dmrs_dict["tokens"][-1]["carg"] = token.carg 
        
        new_node_ids = {}
        for deriv_node_id, deriv_node in self.nodes.items():
            for sem_node in deriv_node.semantic_nodes:
                new_id = len(new_node_ids)
                new_node_ids[sem_node.node_id] = new_id
                dmrs_dict["nodes"].append({"id": new_id, "label": sem_node.original_predicate, "anchors": [{"from": deriv_node.start_token_index, "end": deriv_node.end_token_index}]})

        if dmrs_rep.top is None or dmrs_rep.top not in new_node_ids:
            dmrs_dict["tops"] = []
        else:
            dmrs_dict["tops"] = [new_node_ids[dmrs_rep.top]]

        for edge in dmrs_rep.links:
            if not (edge.start in new_node_ids and edge.end in new_node_ids):
                print(edge.role)
                continue
            dmrs_dict["edges"].append({"source": new_node_ids[edge.start], "target": new_node_ids[edge.end], "label": edge.role, "post-label": edge.post})

        return json.dumps(dmrs_dict)


    def map_dmrs(self, dmrs_rep):
        # Map dmrs nodes to the meaning representation
        for node in dmrs_rep.nodes:
            self.map_dmrs_node(node)

        # Map overlapping nodes
        for node in dmrs_rep.nodes:
            if type(self.dmrs_node_map[node.id]) == tuple:
                self.match_overlapping_dmrs_node(node)
 

    def map_dmrs_node(self, node):
        assert node.lnk.data[0] in self.start_char_token_map and node.lnk.data[1] in self.end_char_token_map, "MRS predicate not matching tokens: " + str(node.original_predicate) + " " + str(node.lnk)
        start_node, end_node = self.start_char_token_map[node.lnk.data[0]], self.end_char_token_map[node.lnk.data[1]]
        start_token, end_token = self.token_nodes[start_node].index, self.token_nodes[end_node].index
        span_str = "%d:%d" % (start_token, end_token)
        matched_node = False

        # match nodes (not token nodes)
        if span_str in self.span_node_map:
            node_id = self.span_node_map[span_str]
            self.nodes[node_id].semantic_nodes.append(SemanticNode(node.id, node.predicate, node.carg))
            self.dmrs_node_map[node.id] = node_id
            matched_node = True
        else:
            self.dmrs_node_map[node.id] = (-1, start_token, end_token)
        
        return matched_node


    def match_overlapping_dmrs_node(self, node):
        # Find matching parent for unmatched node
        start_token, end_token = self.dmrs_node_map[node.id][1], self.dmrs_node_map[node.id][2]
        matched = False

        for node_id, s_node in self.nodes.items():
            if s_node.start_token_index == start_token and s_node.end_token_index == end_token:
                assert s_node.syntax_labels[0] == "NULL" # overlapping node
                self.nodes[node_id].semantic_nodes.append(SemanticNode(node.id, node.predicate, node.carg))
                self.dmrs_node_map[node.id] = node_id
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
                    new_node.semantic_nodes.append(SemanticNode(node.id, node.predicate, node.carg))
                    self.nodes[node_id].overlapping_node_ids.append(new_node_id)
                    self.overlapping_node_map[new_node_id] = node_id
                    self.nodes[new_node_id] = new_node
                    self.dmrs_node_map[node.id] = new_node_id
                    matched = True
                    break

        assert matched, "Not Matched %s %d %d" % (str(node), start_token, end_token) 

    def process_semantic_tree(self, node_id, dmrs_rep, semantic_parent=-1):
        node = self.nodes[node_id]
        sem_node_ids = [snode.node_id for snode in node.semantic_nodes]
        remove_sem_nodes = []
        internal_edge_from = [] # semantic node ids
        internal_edge_to = []
        internal_edge_label = []

        if node.semantic_nodes:
            semantic_anchor = node_id
            node.semantic_parent_node = semantic_parent
            for edge in dmrs_rep.links:
                start_node_id = self.dmrs_node_map[edge.start]
                end_node_id = self.dmrs_node_map[edge.end]
                if end_node_id == node_id:
                    #start_id = sem_node_ids.index(edge.start)
                    end_id = sem_node_ids.index(edge.end)
                    sem_node = node.semantic_nodes[end_id]
                    if start_node_id == node_id:
                        # record internal edge
                        internal_edge_from.append(edge.start) 
                        internal_edge_to.append(edge.end) 
                        internal_edge_label.append(edge.role) 
                        # previously recorded in the node, and test for non-chains
                    elif start_node_id == semantic_parent:
                        # record ancestor edge
                        self.nodes[node_id].semantic_nodes[end_id].has_ancestor = True
                        #assert self.nodes[node_id].semantic_parent_edge_label == ""
                        self.nodes[node_id].semantic_parent_edge_label = edge.role
                        parent_sem_node_ids = [snode.node_id for snode in self.nodes[semantic_parent].semantic_nodes]
                        parent_start_id = parent_sem_node_ids.index(edge.start)
                        self.nodes[semantic_parent].semantic_nodes[parent_start_id].is_semantic_head = True

            # identify non-token-level surface predicates to move
            #   if the node has internal children, don't move
            for sid, sem_node in enumerate(node.semantic_nodes):
                if (not node.isToken) and sem_node.node_id not in internal_edge_from:
                    token_index = -1
                    if d_predicate.is_surface(sem_node.original_predicate):
                        token_index = match_surface_predicate_token(sem_node.original_predicate, node.start_token_index, node.end_token_index, self.token_nodes, self.token_node_list)
                    elif sem_node.carg is not None:
                        token_index = match_surface_predicate_token(sem_node.carg, node.start_token_index, node.end_token_index, self.token_nodes, self.token_node_list)

                    if token_index >= 0:
                        token_id = self.token_node_list[token_index]
                        new_preterminal = self.token_preterminal_node_map[token_id]
                        self.nodes[new_preterminal].semantic_nodes.append(sem_node)
                        self.dmrs_node_map[sem_node.node_id] = new_preterminal
                        remove_sem_nodes.append(sid)
                        # follow the chain
                        # for some quantifiers, might be indended to span everything, but this seems good enough for now
                        snode_id = sem_node.node_id 
                        while snode_id in internal_edge_to:
                            new_snode_id = -1
                            for edge_i, parent_node_id in enumerate(internal_edge_from):
                                if internal_edge_to[edge_i] == snode_id and internal_edge_from.count(parent_node_id) == 1:
                                    sid = sem_node_ids.index(parent_node_id)
                                    sem_node = node.semantic_nodes[sid]
                                    self.nodes[new_preterminal].semantic_nodes.append(sem_node)

                                    self.dmrs_node_map[sem_node.node_id] = new_preterminal
                                    remove_sem_nodes.append(sid)
                                    if parent_node_id in internal_edge_to:
                                        #if new_snode_id >= 0: # almost never have 2 internal parents
                                        new_snode_id = parent_node_id
                            snode_id = new_snode_id

        else:
            semantic_anchor = semantic_parent

        for i in sorted(remove_sem_nodes, reverse=True):
            del node.semantic_nodes[i]
            
        # if current node is an overlapping node and it has nodes left, send to the spanning parent 
        # (if all the arguments of the node is covered by one of the children, should ideally send down, but not now)
        if node.node_id in self.overlapping_node_map and len(node.semantic_nodes) > 0:
            parent_node_id = self.overlapping_node_map[node.node_id]
            for i in range(len(node.semantic_nodes)-1, -1, -1):
                self.nodes[parent_node_id].semantic_nodes.append(node.semantic_nodes[i])
                del node.semantic_nodes[i]

        for child_id in node.overlapping_node_ids:
            self.process_semantic_tree(child_id, dmrs_rep, semantic_anchor)

        # For token (preterminal) nodes, extract lemmas from predicates
        if node.isToken:
            if len(node.token_ids) == 1:
                tok = self.token_nodes[node.token_ids[0]]
                best_lemma_match_prob = 0.0
                best_sid = -1
                best_pred = ""
                t_str = clean_token_lemma(tok.token_str)
                for sid, sem_node in enumerate(node.semantic_nodes):
                    if d_predicate.is_surface(sem_node.original_predicate):
                        lemma, pos, sense = d_predicate.split(sem_node.original_predicate)
                        pred = "_" + ("_".join([pos, sense]) if sense is not None else pos)
                        seq = difflib.SequenceMatcher(a=lemma, b=t_str)
                        lemma_match_prob = seq.ratio()
                        if tok.lemma == "" or lemma_match_prob > best_lemma_match_prob:
                            tok.lemma = lemma
                            best_sid = sid
                            best_pred = pred
                            best_lemma_match_prob = lemma_match_prob
                        if pred == "_u_unknown":
                            if "/" in lemma:
                                tok.lemma = lemma[:lemma.rindex("/")]
                                sem_node.original_predicate = "_" + tok.lemma + pred
                            tok.is_unknown = True
                    if sem_node.carg is not None:
                        if tok.carg == "":
                            tok.carg = sem_node.carg
                        # For multiple CARGs, just take first one as heuristic
                if tok.carg != "":
                    if tok.lemma == "":
                        tok.lemma = tok.carg
                    else:
                        t_str = clean_token_lemma(tok.token_str, True)
                        seq = difflib.SequenceMatcher(a=tok.carg, b=t_str)
                        carg_match_prob = seq.ratio()
                        if carg_match_prob > best_lemma_match_prob:
                            tok.lemma = tok.carg
                            best_lemma_match_prob = carg_match_prob
                #if best_lemma_match_prob < 0.5 and tok.lemma != "" and tok.lemma != tok.carg:
                #    print(tok.lemma, tok.token_str)
                if best_sid >=0 and tok.lemma != tok.carg:
                    node.semantic_nodes[best_sid].predicate = best_pred
            elif len(node.token_ids) > 1:
                matched_multi = False
                for sem_node in node.semantic_nodes:
                    if d_predicate.is_surface(sem_node.original_predicate):
                        lemma, pos, sense = d_predicate.split(sem_node.original_predicate)
                        if "-" in lemma:
                            lemma_split = lemma.split("-")
                            lemma_split[0] += "-"
                        else:
                            lemma_split = lemma.split("+")
                        if len(lemma_split) == len(node.token_ids):
                            pred = "_" + ("_".join([pos, sense]) if sense is not None else pos)
                            sem_node.predicate = pred
                            for i, tok_id in enumerate(node.token_ids):
                                tok = self.token_nodes[tok_id]
                                tok.lemma = lemma_split[i]
                            matched_multi = True
                            break
                    #TODO match the carg if there is one

                if matched_multi:
                    tokstr = [self.token_nodes[tok_id].token_str for tok_id in node.token_ids]
                    semstr = [sem_node.original_predicate for sem_node in node.semantic_nodes]
                    #print("matched", node.token_form, tokstr, semstr)

        for child_id in node.child_node_ids:
            self.process_semantic_tree(child_id, dmrs_rep, semantic_anchor)
 

    def classify_edges(self, dmrs_rep):
        # argument is non-terminal
        for edge in dmrs_rep.links:
            if not self.nodes[self.dmrs_node_map[edge.end]].isToken:
                start_node = self.dmrs_node_map[edge.start]
                start_snode_ids = [snode.node_id for snode in self.nodes[start_node].semantic_nodes]
                start_snode = self.nodes[start_node].semantic_nodes[start_snode_ids.index(edge.start)]
                print(start_snode.predicate, edge.role)
                print(self.nodes[self.dmrs_node_map[edge.start]], self.nodes[self.dmrs_node_map[edge.end]])


def read_profile(args):
    ts = d_itsdb.TestSuite(args.input)
    profile_name = get_profile_name(args.input)
    lexicon = Lexicon(args.grammar)
 
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

        mr = SemanticRepresentation(profile_name + ":" + iid, sentence, token_dict, derivation_rep, lexicon) # read derivation tree

        if args.convert_semantics:
            mr.map_dmrs(dmrs_rep)
            mr.process_semantic_tree(mr.root_node_id, dmrs_rep)

        #TODO method to print out examples

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
 

def read_profile_old(dirname):
    ts = d_itsdb.TestSuite(dirname)
    profile_name = get_profile_name(dirname)

    if True: #TODO print method
        #meaning_representation.classify_edges(dmrs_rep)

        # Print out examples

        print_full = False
        print_overlapping = False
        print_multitokens = False
        print_non_surface = False

        if print_full:
            print(sentence)
            print(meaning_representation.semantic_tree_str(meaning_representation.root_node_id))

        for node_id, node in meaning_representation.nodes.items():
            if print_overlapping and len(node.overlapping_node_ids) > 0:
                 print(meaning_representation.semantic_tree_str(node_id))
            if print_multitokens and len(node.token_ids) > 1:
                print(meaning_representation.semantic_tree_str(node_id))
            if print_non_surface:
                has_surface = False
                for snode in node.semantic_nodes:
                    if (not node.isToken) and d_predicate.is_surface(snode.original_predicate):
                        has_surface = True
                if has_surface:
                    print(meaning_representation.semantic_tree_str(node_id))
        #break

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

