import re
import time
import io
import sys
import argparse
from collections import defaultdict, namedtuple
import math
import pprint
import os

Rule = namedtuple('Rule', 'lhs, rhs')
Terminal = namedtuple('Terminal', 'value')
Nonterminal = namedtuple('Nonterminal', 'value')
Subtree = namedtuple('Subtree', 'parent, children')

'''
reads the parse string by iteratively replacing ( stuff ) with a subtree id. elegant but not efficient.

for example: 
>> ReadParseTree('(S (NP (A good) (N boys)) (VP (V sleep)))')
'''
def ReadParseTree(parse_string):
  rule_matcher = re.compile('(\([^\(\)]+\))')
  placeholder_prefix = '_|+|_'
  nodes = {}
  id = 0
  while parse_string.startswith('('):
    match = rule_matcher.search(parse_string)
    assert match, 'no match!!!'
    parts = match.groups()[0].strip('()').split()
    for i in xrange(len(parts)):
      if parts[i].startswith(placeholder_prefix):
        parts[i] = nodes[parts[i]]
      elif i == 0:
        parts[i] = Nonterminal(value=parts[i])
      else:
        parts[i] = Terminal(value=parts[i])
    new_subtree = Subtree(parent=parts[0], children=parts[1:])
    nodes[placeholder_prefix+str(id)] = new_subtree
    parse_string = parse_string[:match.span()[0]] + placeholder_prefix + str(id) + parse_string[match.span()[1]:]
    id+=1
  assert parse_string.startswith(placeholder_prefix), 'parse string doesnt start with the placeholder'
  return new_subtree

'''
recursively prints out a parse tree. elegant but not efficient

for example: 
>> WriteParseSubtree(Subtree(parent='S', children=[Subtree(parent='NP', children=[Subtree(parent='A', children=['good']), Subtree(parent='N', children=['boys'])]), Subtree(parent='VP', children=[Subtree(parent='V', children=['sleep'])])]))

'( S ( NP ( A good ) ( N boys ) ) ( VP ( V sleep ) ) )'
'''
def WriteParseSubtree(subtree):
  assert subtree is not None, 'subtree is none!'
  if subtree is None:
    exit(1)
  new_parts = ['(', str(subtree.parent.value)]
  for child in subtree.children:
    if type(child) is Terminal:
      new_parts.append(child.value)
    else:
      new_parts.append(WriteParseSubtree(child))
  new_parts.append(')')
  return ' '.join(new_parts)

'''
recursively returns a list of rules used in a subtree
'''
def ExtractRulesFromSubtree(subtree):
  rules = []
  rhs = []
  for child in subtree.children:
    if type(child) is Terminal:
      rhs.append( child )
    else:
      rhs.append( child.parent )
      rules.extend( ExtractRulesFromSubtree(child) )
  rules.append( Rule(lhs=subtree.parent, rhs=rhs) )
  return rules

'''
estimate a PCFG
'''
SOS = 'sentence_boundary'
def EstimatePcfgFromParseTrees(parse_trees):
  # count
  #unique_lhs, unique_rhs = set(), set()
  pcfg = {}
  # pcfg[Nonterminal(SOS)] = {}
  counter = 0
  for subtree in parse_trees:
    counter += 1
    if counter % 1000 == 0:
      print counter, ' trees used to estimate pcfg so far.'
    #unique_lhs.add(Nonterminal(SOS))
    #if Nonterminal(value=subtree.parent) not in pcfg[ Nonterminal(value=SOS) ]:
    #  pcfg[ Nonterminal(value=SOS) ][ Nonterminal(value=subtree.parent) ] = 0
    #pcfg[ Nonterminal(value=SOS) ][ Nonterminal(value=subtree.parent) ] += 1 
    for rule in ExtractRulesFromSubtree(subtree):
      if rule.lhs not in pcfg:
        pcfg[rule.lhs] = {}
      if tuple(rule.rhs) not in pcfg[rule.lhs]:
        pcfg[rule.lhs][tuple(rule.rhs)] = 0
      pcfg[ rule.lhs ][ tuple(rule.rhs) ] += 1.0
  #    unique_lhs.add(rule.lhs)
  #    unique_rhs.add(tuple(rule.rhs))
  # dirichlet prior
  #dirichlet_prior = 0.01
  #for context in unique_lhs:
  #  for decision in unique_rhs:
  #    if decision not in pcfg[context]:
  #      pcfg[context][decision] = 0.0
  #    pcfg[context][decision] += dirichlet_prior
  # normalize
  for context in pcfg:
    context_count = 0.0
    for decision in pcfg[context].keys():
      context_count += pcfg[context][decision]
    for decision in pcfg[context].keys():
      pcfg[context][decision] = math.log(pcfg[context][decision]/context_count)
  return pcfg

'''
convert a parse tree into a pos tagging sequence
assumption: terminal items appear at the rhs of unary rules only
'''
def ConvertParseTreeIntoPosSequence(parse_tree):
  tokens, tags = [], []
  for rule in ExtractRulesFromSubtree(parse_tree):
    if type(rule.rhs[0]) is Terminal:
      tokens.append(rule.rhs[0].value)
      tags.append(rule.lhs.value)
  return (tokens, tags)

'''
estimate an hmm tagger using parse trees
'''
def EstimateHmmPosTaggerFromParseTrees(parse_trees):
  emissions, transitions = {}, {}
  unique_tags, unique_words = set(), set()
  # count
  for parse_tree in parse_trees:
    tokens, tags = ConvertParseTreeIntoPosSequence(parse_tree)
    tags.append(SOS)
    for i in xrange(len(tags)):
      if i != len(tags)-1:
        if tags[i] not in emissions:
          emissions[tags[i]] = {}
        if tokens[i] not in emissions[tags[i]]:
          emissions[tags[i]][tokens[i]] = 0.0
        emissions[tags[i]][tokens[i]] += 1.0
        unique_words.add(tokens[i])
        # end of i != len(tags)
      if tags[i-1] not in transitions:
        transitions[tags[i-1]] = {}
      if tags[i] not in transitions[tags[i-1]]:
        transitions[tags[i-1]][tags[i]] = 0.0
      transitions[tags[i-1]][tags[i]] += 1.0
      unique_tags.add(tags[i])
      # end of processing this position
    #end of processing this parse
  # tag set stats
  print '|tag set| = ', len(unique_tags)
  tagset_file = io.open('postagset', encoding='utf8', mode='w')
  for tag in unique_tags:
    if type(tag) is unicode:
      tagset_file.write(tag)
    elif type(tag) is Nonterminal or type(tag) is Terminal:
      print tag, 'is of type ', type(tag), '!!!' 
  tagset_file.close()
  # add symmetric dirichlet priors for emissions and transitions
  #transitions_dirichlet_alpha = 0.01
  #emissions_dirichlet_alpha = 0.1
  #for context in unique_tags:
  #  for decision in unique_tags:
  #    if decision not in transitions[context]:
  #      transitions[context][decision] = 0.0
  #    transitions[context][decision] += transitions_dirichlet_alpha
  #  if context == Nonterminal(value=SOS):
  #    continue
  #  for decision in unique_words:
  #    if decision not in emissions[context]:
  #      emissions[context][decision] = 0.0
  #    emissions[context][decision] += emissions_dirichlet_alpha
  # normalize
  for distribution in [emissions, transitions]:
    for context in distribution.keys():
      context_count = 0.0
      for decision in distribution[context].keys():
        context_count += distribution[context][decision]
      for decision in distribution[context].keys():
        distribution[context][decision] = math.log(distribution[context][decision]/context_count)
  return (transitions, emissions)

'''
three operations to convert general CFG to CNF: terminal->nonterminal, united-siblings, unary-merge
'''
UNARY_SEPARATOR = u'-unarycollapsed-'
TERMINAL_SEPARATOR = u'terminalinduced-'
UNITED_CHILDREN_SEPARATOR = u'-unitedchildren-'
def ConvertSubtreeIntoChomskyNormalForm(subtree):
  # base case to stop the recursion at CNF tree leaves
  if len(subtree.children) == 1 and type(subtree.children[0]) is Terminal:
    return subtree
  # CNF may have been violated. Lets make local fixes first.
  # one non-terminal child
  if len(subtree.children) == 1 and type(subtree.children[0]) is Subtree:
    new_parent = UNARY_SEPARATOR.join([subtree.parent.value, subtree.children[0].parent.value])
    new_children = subtree.children[0].children
    subtree = Subtree(parent=Nonterminal(new_parent), children=new_children)
    return ConvertSubtreeIntoChomskyNormalForm(subtree)
  # a child needs to be a nonterminal
  if len(subtree.children) > 1:
    for i in xrange(len(subtree.children)):
      if type(subtree.children[i]) is Terminal:
        subtree.children[i] = Subtree(parent=Nonterminal(TERMINAL_SEPARATOR+subtree.children[i].value), \
                                        children = [subtree.children[i]])
  # more than two children
  if len(subtree.children) > 2:
    assert type(subtree.children[0]) is Subtree, 'leftmost child should be a subtree at this point'
    united_children_parent = subtree.children[1].parent.value
    united_children_children = [subtree.children[1]]
    for i in range(2, len(subtree.children)):
      united_children_parent += UNITED_CHILDREN_SEPARATOR + subtree.children[i].parent.value
      united_children_children.append(subtree.children[i])
    united_children = Subtree(parent = Nonterminal(united_children_parent), children = united_children_children)
    subtree.children[1:] = [united_children]
  # now, locally, this subtree should be in CNF
  if len(subtree.children) != 2 or type(subtree.children[0]) is not Subtree or type(subtree.children[1]) is not Subtree:
    print 'parent = ', subtree.parent
    print 'left_child = ', subtree.children[0]
    print 'right_child = ', subtree.children[1]
    pprint.pprint(subtree)
  assert len(subtree.children) == 2 and type(subtree.children[0]) is Subtree and type(subtree.children[1]) is Subtree, \
         'subtree not in cnf'
  # recursively call this method to make sure individual children subtrees are also in CNF
  for i in xrange(len(subtree.children)):
    subtree.children[i] = ConvertSubtreeIntoChomskyNormalForm(subtree.children[i])
  return subtree

'''
convert a CNF parse to the original parse
'''
# THIS IS SCREWED UP
def ConvertCnfSubtreeIntoOriginalSubtree(subtree):
  if subtree.parent.startswith(TERMINAL_SEPARATOR):
    assert len(subtree.children) == 1 and type(subtree.children[0]) is Terminal, 'mysterious ' + TERMINAL_SEPARATOR
    return subtree.children[0]
  if subtree.parent.find(UNARY_SEPARATOR) >= 0:
    united_parents = subtree.parent.split(UNARY_SEPARATOR)
    new_subtree = Subtree(parent = UNARY_SEPARATOR.join(united_parents[0:-1]), \
                            children = Subtree(parent = Nonterminal(united_parents[-1]), children = subtree.children))
    return ConvertCnfSubtreeIntoOriginalSubtree(new_subtree)
  if len(subtree.children) == 2 and subtree.children[1].find(UNITED_CHILDREN_SEPARATOR) >= 0:
    united_children = subtree.children[1]
    subtree.children[1:] = []
    for i in xrange(united_children.children):
      divided_child = Subtree(parent = united_children.children[i].parent, children = united_children.children[i].children)
      subtree.children.append(divided_child)
  for i in xrange(len(subtree.children)):
    if type(subtree.children[i]) is not Terminal:
      subtree.children[i] = ConvertCnfSubtreeIntoOriginalSubtree(subtree.children[i])
  return subtree

'''
takes a list and count how many each element repeats
'''
def CountRuleFrequencies(rules):
  freq = defaultdict(int)
  for rule in rules:
    assert type(rule.rhs) is list, 'rhs should be a list'
    rule_hash = [rule.lhs]
    rule_hash.extend(rule.rhs)
    rule_hash = tuple(rule_hash)
    freq[rule_hash] += 1
  return freq

''' 
returns a tuple: (unnormalized_precision, candidate_rules_count, unnormalized_recall, reference_rules_count)
'''
def EvaluateParseTree(candidate_parse, reference_parse):
  # read candidate/reference rules
  candidate_rules = CountRuleFrequencies( ExtractRulesFromSubtree(candidate_parse) )
  reference_rules = CountRuleFrequencies( ExtractRulesFromSubtree(reference_parse) )
  # compute precision
  candidate_rules_count, unnormalized_precision = 0.0, 0.0
  for rule in candidate_rules:
    candidate_rules_count += candidate_rules[rule]
    unnormalized_precision += min(candidate_rules[rule], reference_rules[rule])
  unnormalized_precision /= candidate_rules_count
  # compute recall
  reference_rules_count, unnormalized_recall = 0.0, 0.0
  for rule in reference_rules:
    reference_rules_count += reference_rules[rule]
    unnormalized_recall += min(candidate_rules[rule], reference_rules[rule])
  unnormalized_recall /= reference_rules_count
  return (unnormalized_precision, candidate_rules_count, unnormalized_recall, reference_rules_count)

'''
returns a tuple (correct, all)
'''
def EvaluatePos(candidate_postags, reference_postags):  
  if len(candidate_postags) != len(reference_postags):
    return (0, len(reference_postags))
  correct = 0
  for i in xrange(len(candidate_postags)):
    if candidate_postags[i] == reference_postags[i]:
      correct += 1
  return (correct, len(reference_postags))

'''
given a complete table
'''
def CykBacktrack(table, index):
  assert type(table[index].lhs) is Nonterminal
  subtree = Subtree(parent = table[index].lhs, children = [])
  # basecase
  if table[index].leftchild_index is None:
    subtree.children.append( table[index].rhs[0] )
    return subtree
  subtree.children.append(CykBacktrack(table, table[index].leftchild_index))
  subtree.children.append(CykBacktrack(table, table[index].rightchild_index))
  return subtree

'''
cyk
table maps a tuple (length, start, nonterminal) to CykTuple
'''
def CykParse(tokens, pcfg, start_symbol):
  CykTuple = namedtuple('CykTuple', 'lhs, rhs, logprob, leftchild_index, rightchild_index')
  table = {}
  for length in range(1, len(tokens)+1):
    for start in range(0, len(tokens)-length+1):
      # find a rule that matches this token
      if length == 1:
        for lhs in pcfg.keys():
          for rhs in pcfg[lhs].keys():
            if len(rhs) == 1 and type(rhs[0]) is Terminal and rhs[0].value == tokens[start]:
              table[(length, start, lhs)] = CykTuple(lhs=lhs, rhs=rhs, logprob=pcfg[lhs][rhs], leftchild_index=None, rightchild_index=None)
        # end of length == 1
      # for each possible split point
      for split in range(start+1, start+length):
        for lhs in pcfg.keys():
          for rhs in pcfg[lhs].keys():
            if len(rhs) == 1: continue
            leftchild_index, rightchild_index = (split - start, start, rhs[0]), (length - split + start, split, rhs[1])
            if len(rhs) == 2 and leftchild_index in table and rightchild_index in table:
              match = CykTuple(lhs=lhs, rhs=rhs, \
                                 logprob = table[leftchild_index].logprob + table[rightchild_index].logprob + pcfg[lhs][rhs], \
                                 leftchild_index=leftchild_index, \
                                 rightchild_index=rightchild_index)
              if (length, start, lhs) not in table or table[(length, start, lhs)].logprob < match.logprob:
                table[ (length, start, lhs) ] = match
            # done processing this rule
          # done processing this lhs
        # done processing this potential split point
      # done processing this cell in the cyk table
    # done processing this row in the cyk table
  # done processing all rows in the cyk table, time to backtrack (if any complete parse is available)
  if (len(tokens), 0, Nonterminal(start_symbol)) not in table:
    return None
  return CykBacktrack(table, (len(tokens), 0, Nonterminal(start_symbol)))

def HmmViterbi(tokens, transitions, emissions):
  alpha = {}
  AlphaIndex = namedtuple('AlphaIndex', 'position, state')
  AlphaValue = namedtuple('AlphaValue', 'logprob, prev_state')
  alpha[ AlphaIndex(position=-1, state=Nonterminal(SOS)) ] = AlphaValue(logprob=0, prev_state=None) # you are at the start-of-sent state with prob 1 at the beginning
  for position in range(0, len(tokens)):
    for current_state in transitions.keys():
      for previous_state in transitions.keys():
        if AlphaIndex(position=position-1, state=previous_state) not in alpha or \
              current_state not in transitions[previous_state] or \
              current_state not in emissions or \
              Terminal(tokens[position]) not in emissions[current_state]:
          continue  
        forward_score = alpha[ AlphaIndex(position=position-1, state=previous_state) ].logprob + \
            transitions[previous_state][current_state] + \
            emissions[current_state][Terminal(tokens[position])]
        if AlphaIndex(position=position, state=current_state) not in alpha or \
              alpha[AlphaIndex(position=position, state=current_state)] < AlphaValue(logprob=forward_score, prev_state=previous_state):
          alpha[AlphaIndex(position=position, state=current_state)] = \
              AlphaValue(logprob=forward_score, prev_state=previous_state)
        # done considering a particular previous state
      # done considering a particular current state
    # done processing a particular position
  # reached the end of the sentence, but haven't considered the transition to end-of-sentence state
  # add alpha entries which take into consideration end of sentence boundaries
  for key in alpha.keys():
    if key.position != len(tokens)-1 or Nonterminal(SOS) not in transitions[key.state]:
      continue
    forward_score = alpha[key].logprob + transitions[key.state][Nonterminal(SOS)]
    if AlphaIndex(position=len(tokens), state=Nonterminal(SOS)) not in alpha or \
          alpha[ AlphaIndex(position=len(tokens), state=Nonterminal(SOS)) ] < AlphaValue(logprob=forward_score, prev_state=key.state):
      alpha[ AlphaIndex(position=len(tokens), state=Nonterminal(SOS)) ] = AlphaValue(logprob=forward_score, prev_state=key.state)
  # backtrack
  current_state = Nonterminal(SOS)
  viterbi_tag_sequence = []
  for position in reversed(range(1, len(tokens)+1)):
    if AlphaIndex(position, current_state) not in alpha:
      return None
    prev_state = alpha[ AlphaIndex(position, current_state) ].prev_state
    viterbi_tag_sequence.insert(0, prev_state)
    current_state = prev_state
  return viterbi_tag_sequence

def ReadTreebankDir(treebank_dir):
  parses = []
  for filename in os.listdir(treebank_dir):
    filename = os.path.join(treebank_dir, filename)
    if not filename.endswith('.tree'): continue
    for line in io.open(filename, encoding='utf8'):
      parses.append(ReadParseTree(line.strip()))
  return parses


# parse/validate arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("--create_hw", type=bool, default=True)
argParser.add_argument("--solve_hw", type=bool, default=True)
argParser.add_argument("--treebank_dir", type=str, default='/usr1/home/wammar/atb3_v3_2/data/penntree/without-vowel/')
argParser.add_argument("--dev_sents_filename", type=str, default='dev_sents')
argParser.add_argument("--dev_parses_filename", type=str, default='dev_parses')
argParser.add_argument("--dev_postags_filename", type=str, default='dev_postags')
argParser.add_argument("--test_sents_filename", type=str, default='test_sents')
argParser.add_argument("--test_parses_filename", type=str, default='test_parses')
argParser.add_argument("--test_postags_filename", type=str, default='test_postags')
argParser.add_argument("--train_sents_filename", type=str, default='train_sents')
argParser.add_argument("--train_parses_filename", type=str, default='train_parses')
argParser.add_argument("--train_postags_filename", type=str, default='train_postags')
argParser.add_argument("--hmm_transitions_filename", type=str, default='hmm_trans')
argParser.add_argument("--hmm_emissions_filename", type=str, default='hmm_emits')
argParser.add_argument("--pcfg_filename", type=str, default='pcfg')
argParser.add_argument("--candid_dev_parses_filename", type=str, default='candid_dev_parses')
argParser.add_argument("--candid_dev_postags_filename", type=str, default='candid_dev_postags')
argParser.add_argument("--candid_test_parses_filename", type=str, default='candid_test_parses')
argParser.add_argument("--candid_test_postags_filename", type=str, default='candid_test_postags')
args = argParser.parse_args()


if args.create_hw:
  # read all of the Arabic treebank
  parses = ReadTreebankDir(args.treebank_dir)
  print len(parses), ' parses read.'
  
  # each ninth and tenth example will be put in a dev and test set
  tree_id = 0
  dev_sents_file = io.open(args.dev_sents_filename, encoding='utf8', mode='w')
  dev_parses_file = io.open(args.dev_parses_filename, encoding='utf8', mode='w')
  dev_postags_file = io.open(args.dev_postags_filename, encoding='utf8', mode='w')
  test_sents_file = io.open(args.test_sents_filename, encoding='utf8', mode='w')
  test_parses_file = io.open(args.test_parses_filename, encoding='utf8', mode='w')
  test_postags_file = io.open(args.test_postags_filename, encoding='utf8', mode='w')
  train_sents_file = io.open(args.train_sents_filename, encoding='utf8', mode='w')
  train_parses_file = io.open(args.train_parses_filename, encoding='utf8', mode='w')
  train_postags_file = io.open(args.train_postags_filename, encoding='utf8', mode='w')
  train, dev, test = [], [], []
  train_sents, dev_sents, test_sents = [], [], []
  for tree in parses:
    if tree.parent != Nonterminal('S'): continue
    # convert trees into chomsky normal form
    tree = ConvertSubtreeIntoChomskyNormalForm(tree)
    (tokens, tags) = ConvertParseTreeIntoPosSequence(tree)
    sent = u'{0}\n'.format(' '.join(tokens))
    parse_string = u'{0}\n'.format(WriteParseSubtree(tree))
    #print parse_string
    postags_string = u'{0}\n'.format(' '.join(tags))
    # distribute on train, dev, test
    if tree_id % 10 == 0:
      dev.append(tree)
      dev_sents_file.write(sent)
      dev_sents.append(sent)
      dev_parses_file.write(parse_string)
      dev_postags_file.write(postags_string)
    elif tree_id % 10 == 1:
      test.append(tree)
      test_sents_file.write(sent)
      test_sents.append(sent)
      test_parses_file.write(parse_string)
      test_postags_file.write(postags_string)
    else:
      train.append(tree)
      train_sents_file.write(sent)
      train_sents.append(sent)
      train_parses_file.write(parse_string)
      train_postags_file.write(postags_string)
    tree_id += 1
  dev_sents_file.close()
  dev_parses_file.close()
  dev_postags_file.close()
  test_sents_file.close()
  test_parses_file.close()
  test_postags_file.close()
  train_sents_file.close()
  train_parses_file.close()
  train_postags_file.close()

  # estimate hmm model
  (transitions, emissions) = EstimateHmmPosTaggerFromParseTrees(train)
  hmm_transitions_file = io.open(args.hmm_transitions_filename, encoding='utf8', mode='w')
  for context in transitions.keys():
    for decision in transitions[context].keys():
      hmm_transitions_file.write(u'{0}\t{1}\t{2}\n'.format(context.strip(), decision.strip(), transitions[context][decision]))
  hmm_transitions_file.close()
  hmm_emissions_file = io.open(args.hmm_emissions_filename, encoding='utf8', mode='w')
  for context in emissions.keys():
    for decision in emissions[context].keys():
      hmm_emissions_file.write(u'{0}\t{1}\t{2}\n'.format(context.strip(), decision.strip(), emissions[context][decision]))
  hmm_emissions_file.close()

  # estimate pcfg
  pcfg = EstimatePcfgFromParseTrees(train)
  pcfg_file = io.open(args.pcfg_filename, encoding='utf8', mode='w')
  for context in pcfg.keys():
    for decision in pcfg[context].keys():
      decision_string = ''
      for i in xrange(len(decision)):
        decision_string += decision[i].value + u' '
      pcfg_file.write(u'{0}\t{1}\t{2}\n'.format(context.value, decision_string.strip(), pcfg[context][decision]))
  pcfg_file.close()
  pcfg_file.close()

if args.solve_hw:
  print 'now decoding the dev set...'
  # use the pcfg models to parse something
  # use the hmm model to parse something
  candid_dev_parses_file = io.open(args.candid_dev_parses_filename, encoding='utf8', mode='w')
  candid_dev_postags_file = io.open(args.candid_dev_postags_filename, encoding='utf8', mode='w')
  parse_failures, tagging_failures = 0, 0
  for i in xrange(len(dev_sents)):
    sent = dev_sents[i]
    # parse
    tree = CykParse(sent.split(), pcfg, u'S')
    if tree is None:
      parse_failures += 1
      candid_dev_parses_file.write(u'\n')
    else:
      candid_dev_parses_file.write(u'{0}\n'.format(WriteParseSubtree(tree)))
    # tag
    tagging = HmmViterbi(sent.split(), transitions, emissions)
    if tagging is None:
      tagging_failures += 1
      candid_dev_postags_file.write(u'\n')
    else:
      candid_dev_postags_file.write(u'{0}\n', ' '.join(tagging))
  print tagging_failures, ' failures tagging ', len(dev_sents), ' sents'
  print parse_failures, ' failures cyk parsing ', len(dev_sents), ' sents' 
  candid_dev_parses_file.close()
  candid_dev_postags_file.close()

#ref_tree = ReadParseTree('(S (NP (ADJ good) (N boys)) (VP sleep))')
#can_tree = ReadParseTree('(S (NP (ADJ good) (N boys)) (VP sleep2))')
#print 'raw ref_tree is ', ref_tree
#print 'ref_tree is ', WriteParseSubtree(ref_tree), '\n'
#print 'can_tree is ', WriteParseSubtree(can_tree), '\n'

# evaluation
#unnormalized_precision, can_rules_count, unnormalized_recall, ref_rules_count = EvaluateParseTree(can_tree, ref_tree)
#print 'precision=', unnormalized_precision / can_rules_count, ', recall=', unnormalized_recall / ref_rules_count
#(ref_toks, ref_pos), (can_toks, can_pos) = ConvertParseTreeIntoPosSequence(ref_tree), ConvertParseTreeIntoPosSequence(can_tree)
#print 'ref_pos = ', ref_pos
#print 'can_pos = ', can_pos
#correct, all = EvaluatePos(can_pos, ref_pos)
#print correct, ' correct POS out of ', all

# estimation
#transitions, emissions = EstimateHmmPosTaggerFromParseTrees([ref_tree, can_tree])
#pcfg = EstimatePcfgFromParseTrees([ref_tree, can_tree])
#print '\npcfg='
#for c in pcfg.keys():
#  for d in pcfg[c].keys():
#    print d,'|',c,'=',pcfg[c][d]

# cyk
#print 'now running cyk\n'
#parse = CykParse(['sleep', 'boys', 'good'], pcfg, 'S')
#print '\n raw parse = ', parse
#print '\n nicely drawn parse=', WriteParseSubtree(parse)

# hmm viterbi
#print '\n now running hmm viterbi'
#viterbi = HmmViterbi(['good', 'boys', 'sleep'], transitions, emissions)
#print viterbi

#cnf = ConvertSubtreeIntoChomskyNormalForm(tree)
#print
#print WriteParseSubtree(cnf)

