from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf


def model_fn(features, targets, mode, params, scope=None):
    embedding_size = params['embedding_size']
    vocab_size = params['vocab_size']
    hidden_units = params['hidden_units']
    debug = params['debug']

    story = features['story']
    query = features['query']

    batch_size = tf.shape(story)[0]

    normal_initializer = tf.random_normal_initializer(stddev=0.1)
    ones_initializer = tf.constant_initializer(1.0)

    with tf.variable_scope(scope, 'Entity_Network', initializer=normal_initializer):
        # Embeddings taking care of paddings
        embedding_params = tf.get_variable('embedding_params', [vocab_size, embedding_size])
        embedding_mask = tf.constant([0 if i == 0 else 1 for i in range(vocab_size)],
                                     dtype=tf.float32,
                                     shape=[vocab_size, 1])
        embedding_params_masked = embedding_params * embedding_mask

        story_embedding = tf.nn.embedding_lookup(embedding_params_masked, story)
        query_embedding = tf.nn.embedding_lookup(embedding_params_masked, query)
        # Input Module
        encoded_story = get_input_encoding(story_embedding, ones_initializer, 'StoryEncoding')
        encoded_query = get_input_encoding(query_embedding, ones_initializer, 'QueryEncoding')

        # Recurrence
        cell = tf.nn.rnn_cell.GRUCell(hidden_units)
        initial_state = cell.zero_state(batch_size, tf.float32)
        story_length = get_story_length(encoded_story)

        _, last_state = tf.nn.dynamic_rnn(cell, encoded_story,
                                          sequence_length=story_length,
                                          initial_state=initial_state)

        # Output Module
        output = get_output(last_state,
                            vocab_size=vocab_size,
                            initializer=normal_initializer
                            )
        prediction = tf.argmax(output, 1)

        # Training
        loss = get_loss(output, targets, mode)
        train_op = training_optimizer(loss, params, mode)
        if debug:
            tf.contrib.layers.summarize_tensor(story_length, 'story_length')
            tf.contrib.layers.summarize_tensor(encoded_story, 'encoded_story')
            tf.contrib.layers.summarize_tensor(encoded_query, 'encoded_query')
            tf.contrib.layers.summarize_tensor(last_state, 'last_state')
            tf.contrib.layers.summarize_tensor(output, 'output')
            tf.contrib.layers.summarize_variables()

            tf.add_check_numerics_ops()

        return prediction, loss, train_op
