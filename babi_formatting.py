"""
This module loads and pre-processes the bAbI dataset [v1.2] into TFRecords.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import re
import json
import tarfile
import tensorflow as tf

from tqdm import tqdm

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('source_dir', 'datasets/', 'Directory containing bAbI sources.')
tf.app.flags.DEFINE_string('dest_dir', 'datasets/processed/', 'Where to write datasets.')
tf.app.flags.DEFINE_boolean('include_10k', False, 'Whether to use 10k or 1k examples.')

SPLIT_RE = re.compile('(\W+)?')

PAD_TOKEN = '_PAD'
PAD_ID = 0


def tokenize_char(sentence):
    """
    Tokenize a string by splitting on characters.
    """
    return list(sentence.lower())

def tokenize(sentence):
    """
    Tokenize a string by splitting on non-word characters and stripping whitespace.
    """
    return [token.strip().lower() for token in re.split(SPLIT_RE, sentence) if token.strip()]

def parse_stories(lines, only_supporting=False):
    """
    Parse the bAbI task format.
    If only_supporting is True, only the sentences that support the answer are kept.
    """
    stories = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            query, answer, supporting = line.split('\t')
            query = tokenize(query)
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            stories.append((substory, query, answer))
            story.append('')
        else:
            sentence = tokenize_char(line)
            story.append(sentence)
    return stories


def truncate_stories(stories, max_length):
    stories_truncated = []
    for story, query, answer in stories:
        story_truncated = story[-max_length:]
        stories_truncated.append((story_truncated, query, answer))
    return stories_truncated

def get_tokenizer(stories):
    """
    Recover unique tokens as a vocab and map the tokens to ids.
    """
    tokens_all = []
    for story, query, answer in stories:
        tokens_all.extend([token for sentence in story for token in sentence] + query + [answer])
    vocab = [PAD_TOKEN] + sorted(set(tokens_all))
    token_to_id = {token: i for i, token in enumerate(vocab)}
    return token_to_id

def tokenize_stories(stories, token_to_id):
    """
    Convert all tokens into their unique ids.
    """
    story_ids = []
    for story, query, answer in stories:
        story = [[token_to_id[token] for token in sentence] for sentence in story]
        query = [token_to_id[token] for token in query]
        answer = token_to_id[answer]
        story_ids.append((story, query, answer))
    return story_ids

def pad_stories(stories, max_sentence_length, max_story_length, max_query_length):
    """
    Pad sentences, stories, and queries to a consistence length.
    """
    for story, query, answer in stories:
        for sentence in story:
            for _ in range(max_sentence_length - len(sentence)):
                sentence.append(PAD_ID)
            assert len(sentence) == max_sentence_length

        for _ in range(max_story_length - len(story)):
            story.append([PAD_ID for _ in range(max_sentence_length)])

        for _ in range(max_query_length - len(query)):
            query.append(PAD_ID)

        assert len(story) == max_story_length
        assert len(query) == max_query_length

    return stories

def int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
 
