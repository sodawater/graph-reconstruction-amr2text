from modules import *
import math
from tensorflow.python.layers import core as layers_core
from data_utils import *
import random


class GraphTransformer():
    def __init__(self, hparams, mode):
        self.hparams = hparams
        self.wvocab_size = hparams.from_vocab_size
        self.evocab_size = hparams.edge_vocab_size * 2
        self.num_units = hparams.num_units
        self.emb_dim = hparams.emb_dim
        self.enc_layers = hparams.enc_layers
        self.dec_layers = hparams.dec_layers
        self.num_heads = hparams.num_heads
        self.use_copy = hparams.use_copy
        self.use_charlstm = hparams.use_charlstm
        self.use_depth = hparams.use_depth
        self.learning_rate = tf.Variable(float(hparams.learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * 0.5)
        self.clip_value = hparams.clip_value
        self.dropout_rate = hparams.dropout_rate
        self.max_src_length = hparams.max_src_len
        self.max_tgt_length = hparams.max_tgt_len
        self.beam_width = 5
        self.init_weight = hparams.init_weight
        self.flag = True
        self.mode = mode
        self.batch_size = hparams.batch_size
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.is_training = True
        else:
            self.is_training = False

        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            self.enc_seq_ids = tf.placeholder(tf.int32, [None, self.max_src_length])
            self.enc_seq_lens = tf.placeholder(tf.int32, [None])
            self.enc_seq_masks = tf.placeholder(tf.int32, [None, None, None])
            self.enc_seq_masks_rev = tf.placeholder(tf.int32, [None, None, None])
            self.enc_depth = tf.placeholder(tf.int32, [None, self.max_src_length])
            self.enc_edge_ids = tf.placeholder(tf.int32, [None, self.max_src_length])
            self.enc_edge_links1 = tf.placeholder(tf.int32, [None, self.max_src_length])
            self.enc_edge_links2 = tf.placeholder(tf.int32, [None, self.max_src_length])
            self.recon_edge_distids1 = tf.placeholder(tf.int32, [None, self.max_src_length * 1])
            self.recon_edge_distids2 = tf.placeholder(tf.int32, [None, self.max_src_length * 1])
            self.recon_edge_dists = tf.placeholder(tf.int32, [None, self.max_src_length * 1])
            self.recon_edge_neglinks2 = tf.placeholder(tf.int32, [None, self.max_src_length * 1])
            self.recon_edge_neglinks1 = tf.placeholder(tf.int32, [None, self.max_src_length * 1])
            self.enc_char_ids = tf.placeholder(tf.int32, [None, self.max_src_length, 20])
            self.enc_char_lens = tf.placeholder(tf.int32, [None, self.max_src_length])
            self.enc_edge_lens = tf.placeholder(tf.int32, [None])
            self.dec_ids = tf.placeholder(tf.int32, [None, self.max_tgt_length])
            self.dec_lens = tf.placeholder(tf.int32, [None])
            self.targets = tf.placeholder(tf.int32, [None, None])
            self.weights = tf.placeholder(tf.float32, [None, None])
            self.dec_masks = tf.placeholder(tf.int32, [None, None, None])
        else:
            self.enc_seq_ids = tf.placeholder(tf.int32, [None, self.max_src_length])
            self.enc_seq_lens = tf.placeholder(tf.int32, [None])
            self.enc_depth = tf.placeholder(tf.int32, [None, self.max_src_length])
            self.enc_seq_masks = tf.placeholder(tf.int32, [None, None, None])
            self.enc_seq_masks_rev = tf.placeholder(tf.int32, [None, None, None])
            self.enc_edge_ids = tf.placeholder(tf.int32, [None, self.max_src_length])
            self.enc_char_ids = tf.placeholder(tf.int32, [None, self.max_src_length, 20])
            self.enc_char_lens = tf.placeholder(tf.int32, [None, self.max_src_length])
            self.enc_edge_links1 = tf.placeholder(tf.int32, [None, self.max_src_length])
            self.enc_edge_links2 = tf.placeholder(tf.int32, [None, self.max_src_length])
            self.enc_edge_lens = tf.placeholder(tf.int32, [None])
            self.recon_edge_distids1 = tf.placeholder(tf.int32, [None, self.max_src_length * 1])
            self.recon_edge_distids2 = tf.placeholder(tf.int32, [None, self.max_src_length * 1])
            self.recon_edge_dists = tf.placeholder(tf.int32, [None, self.max_src_length * 1])
            self.recon_edge_neglinks2 = tf.placeholder(tf.int32, [None, self.max_src_length * 1])
            self.recon_edge_neglinks1 = tf.placeholder(tf.int32, [None, self.max_src_length * 1])
            self.dec_ids = tf.placeholder(tf.int32, [None, self.max_tgt_length])
            self.dec_lens = tf.placeholder(tf.int32, [None])
            self.dec_masks = tf.placeholder(tf.int32, [None, None, None])

        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.input_keep_prob = 1 - self.dropout_rate
            self.output_keep_prob = 1.0
        else:
            self.input_keep_prob = 1.0
            self.output_keep_prob = 1.0

        with tf.variable_scope("embedding") as scope:
           
            self.src_embeddings = tf.Variable(hparams.embeddings, trainable=True)
            self.tgt_embeddings = self.src_embeddings  # tf.Variable(hparams.embeddings, trainable=True)
            self.edge_embeddings = tf.Variable(self.init_matrix([self.evocab_size, self.num_units]), trainable=True)
          
            self.edge_embeddings_rev = tf.Variable(self.init_matrix([self.evocab_size, self.num_units]), trainable=True)
            self.edge_embeddings = tf.concat([self.edge_embeddings, self.edge_embeddings_rev], axis=0)

        self.bias_w = tf.Variable(0.1, dtype=tf.float32, trainable=True)

        with tf.variable_scope("project"):
            self.output_layer = layers_core.Dense(self.wvocab_size, use_bias=True)
            self.input_layer1 = layers_core.Dense(self.num_units, use_bias=False)
            self.input_layer2 = layers_core.Dense(self.num_units, use_bias=False)

            
       
        with tf.variable_scope("encoder") as scope:
            self.enc_word_emb = self.input_layer1(tf.nn.embedding_lookup(self.src_embeddings, self.enc_seq_ids))
            
            self.enc_seq_embs = self.enc_word_emb  
            if self.is_training:
                seq_inputs = tf.layers.dropout(self.enc_seq_embs, rate=self.dropout_rate)
            else:
                seq_inputs = self.enc_seq_embs
            self.enc_seq_embs = seq_inputs
            seq_inputs_rev = seq_inputs
            self.enc_edge_emb = tf.nn.embedding_lookup(self.edge_embeddings,
                                                       self.enc_edge_ids)  
            self.enc_edgerev_emb = tf.nn.embedding_lookup(self.edge_embeddings_rev,
                                                          self.enc_edge_ids)  
           
            self.idx_pairs1 = index_matrix_to_pairs(self.enc_edge_links1)
            self.idx_pairs2 = index_matrix_to_pairs(self.enc_edge_links2)


            self.idx_pairs_dists = index_matrix_to_pairs(self.recon_edge_dists)
            self.idx_pairs_neg2 = index_matrix_to_pairs(self.recon_edge_neglinks2)
            self.idx_pairs_neg1 = index_matrix_to_pairs(self.recon_edge_neglinks1)

            self.idx_pairs_dist1 = index_matrix_to_pairs(self.recon_edge_distids1)
            self.idx_pairs_dist2 = index_matrix_to_pairs(self.recon_edge_distids2)
            
            enc_states = []
            enc_states_rev = []
            enc_states.append(seq_inputs)
            enc_states_rev.append(seq_inputs_rev)

            for i in range(self.enc_layers):
                with tf.variable_scope("num_layers_{}".format(i), reuse=tf.AUTO_REUSE):
                    link1 = tf.gather_nd(seq_inputs, self.idx_pairs1)
                    link2 = tf.gather_nd(seq_inputs, self.idx_pairs2)
                  
                    pairs = tf.concat([link1, self.enc_edge_emb, link2], axis=2)
                   
                    pairs_rev = tf.concat([link2, self.enc_edgerev_emb, link1], axis=2)

                    keys = tf.concat([self.enc_edge_emb, link2], axis=2)
                    
                    keys_rev = tf.concat([self.enc_edgerev_emb, link1], axis=2)
                    

                    seq_outputs = multihead_attention_edge(queries=seq_inputs,
                                                           keys=pairs,
                                                           edges=self.enc_edge_emb,
                                                           pairs=keys,
                                                           query_length=self.enc_seq_lens,
                                                           key_length=self.enc_edge_lens,
                                                           num_units=self.num_units,
                                                           num_heads=self.num_heads,
                                                           dropout_rate=self.dropout_rate,
                                                           is_training=self.is_training,
                                                           using_mask=True,
                                                           mymasks=self.enc_seq_masks,
                                                           # mymasks=enc_seq_masks,
                                                           scope="graph_attention1",
                                                           # no_att=True,
                                                           )

                    seq_outputs_rev = multihead_attention_edge(queries=seq_inputs,
                                                               keys=pairs_rev,
                                                               edges=self.enc_edgerev_emb,
                                                               pairs=keys_rev,
                                                               query_length=self.enc_seq_lens,
                                                               key_length=self.enc_edge_lens,
                                                               num_units=self.num_units,
                                                               num_heads=self.num_heads,
                                                               dropout_rate=self.dropout_rate,
                                                               is_training=self.is_training,
                                                               using_mask=True,
                                                               mymasks=self.enc_seq_masks_rev,
                                                               scope="graph_attention2",
                                                               # no_att=True,
                                                               )
                    outputs = tf.concat([seq_outputs, seq_outputs_rev, seq_inputs], axis=2)
                    seq_outputs = tf.layers.dense(outputs, self.num_units, use_bias=True)

                   

                    seq_outputs = feedforward(seq_outputs, [self.num_units * 4, self.num_units * 1],
                                              is_training=self.is_training, dropout_rate=self.dropout_rate, scope="ff1")
                    seq_outputs = normalize(seq_outputs + seq_inputs)
                    seq_inputs = seq_outputs
                   

                    enc_states.append(seq_inputs)
                    enc_states_rev.append(seq_inputs_rev)

            

            seq_outputs = tf.concat(enc_states, axis=2)
            seq_outputs = tf.layers.dense(seq_outputs, self.num_units, use_bias=True, activation=None)

           


            link1 = tf.gather_nd(seq_outputs, self.idx_pairs1)
            link2 = tf.gather_nd(seq_outputs, self.idx_pairs2)
            link_neg2 = tf.gather_nd(seq_outputs, self.idx_pairs_neg2)
            link_neg1 = tf.gather_nd(seq_outputs, self.idx_pairs_neg1)
            link_dist1 = tf.gather_nd(seq_outputs, self.idx_pairs_dist1)
            link_dist2 = tf.gather_nd(seq_outputs, self.idx_pairs_dist2)
            
        with tf.variable_scope("decoder") as scope:
            self.dec_word_emb = self.input_layer1(tf.nn.embedding_lookup(self.src_embeddings, self.dec_ids))
            self.dec_pos_emb = positional_encoding(self.max_tgt_length, self.dec_ids, self.num_units, zero_pad=False,
                                                   scale=False)

            if self.is_training:
                inputs = tf.layers.dropout(self.dec_word_emb + self.dec_pos_emb, rate=self.dropout_rate)
            else:
                inputs = self.dec_word_emb + self.dec_pos_emb


            

            dec_states = []
            for i in range(self.dec_layers):
                with tf.variable_scope("num_layers_{}".format(i), reuse=tf.AUTO_REUSE):
                   
                    outputs = multihead_attention(queries=inputs,
                                                  keys=inputs,
                                                  query_length=self.dec_lens,
                                                  key_length=self.dec_lens,
                                                  num_units=self.num_units,
                                                  num_heads=self.num_heads,
                                                  dropout_rate=self.dropout_rate,
                                                  is_training=self.is_training,
                                                  using_mask=True,
                                                  mymasks=self.dec_masks,
                                                  scope="self_attention")

        
                    outputs = outputs + inputs
                    inputs = normalize(outputs)


                    outputs, weights = multihead_attention(queries=inputs,
                                                           keys=seq_outputs,
                                                           query_length=self.dec_lens,
                                                           key_length=self.enc_seq_lens,
                                                           num_units=self.num_units,
                                                           num_heads=self.num_heads,
                                                           dropout_rate=self.dropout_rate,
                                                           is_training=self.is_training,
                                                           using_mask=False,
                                                           no_att=True,
                                                           scope="multi_head_attention1")

                    outputs = outputs + inputs
                    inputs = normalize(outputs)



                    outputs = feedforward(inputs, [self.num_units * 4, self.num_units], is_training=self.is_training,
                                          dropout_rate=self.dropout_rate, scope="ff")
        
                    outputs = outputs + inputs
                    inputs = normalize(outputs)
                    dec_states.append(inputs)

            decoder_outputs = inputs

        with tf.variable_scope("output", reuse=tf.AUTO_REUSE) as scope:
            if self.use_copy:
                self.generate_logits = self.output_layer(decoder_outputs)
                self.generate_probs = tf.nn.softmax(self.generate_logits)
                
                self.label_logits = feedforward(tf.concat([link1, link2], axis=2), [self.num_units, self.num_units], is_training=self.is_training, activation=gelu,
                                          dropout_rate=self.dropout_rate, scope="ff")
                self.label_logits = tf.layers.dense(self.label_logits, self.evocab_size,
                                                    name="f1")
               

                self.dist_logits = feedforward(tf.concat([link_dist1, link_dist2], axis=2), [self.num_units, self.num_units],
                                               is_training=self.is_training, activation=gelu,
                                               dropout_rate=self.dropout_rate, scope="fff")
                self.dist_logits = tf.layers.dense(self.dist_logits, 14,
                                                   name="f2")
                self.neg1_label_logits = feedforward(tf.concat([link1, link_neg2], axis=2), [self.num_units, self.num_units], is_training=self.is_training, activation=gelu,
                                          dropout_rate=self.dropout_rate, scope="ff")
                self.neg1_label_logits = tf.layers.dense(self.neg1_label_logits, self.evocab_size, name="f1")
        
                self.neg2_label_logits = feedforward(tf.concat([link_neg1, link2], axis=2), [self.num_units , self.num_units], is_training=self.is_training, activation=gelu,
                                          dropout_rate=self.dropout_rate, scope="ff")    
                self.neg2_label_logits = tf.layers.dense(self.neg2_label_logits, self.evocab_size, name="f1")
                
                copy = weights

                link_inputs1 = feedforward(tf.concat([link1, self.enc_edgerev_emb], axis=2), [self.num_units, self.num_units], is_training=self.is_training, activation=gelu,
                                          dropout_rate=self.dropout_rate, scope="ff1")

                link_inputs2 = feedforward(tf.concat([link2, self.enc_edge_emb], axis=2), [self.num_units, self.num_units], is_training=self.is_training, activation=gelu,
                                          dropout_rate=self.dropout_rate, scope="ff1")

                self.link_logits1 = multihead_attention(queries=link_inputs1,
                                                        keys=seq_outputs,
                                                        query_length=self.enc_edge_lens,
                                                        key_length=self.enc_seq_lens,
                                                        num_heads=1,
                                                        num_units=self.num_units,
                                                        dropout_rate=self.dropout_rate,
                                                        is_training=self.is_training,
                                                        logits=True,
                                                        scope="pointer")
                self.link_logits2 = multihead_attention(queries=link_inputs2,
                                                        keys=seq_outputs,
                                                        query_length=self.enc_edge_lens,
                                                        key_length=self.enc_seq_lens,
                                                        num_heads=1,
                                                        num_units=self.num_units,
                                                        dropout_rate=self.dropout_rate,
                                                        is_training=self.is_training,
                                                        logits=True,
                                                        scope="pointer2")
               
                self.copy_or_generate = tf.layers.dense(tf.concat([decoder_outputs, self.dec_word_emb], axis=2), self.num_units, activation=tf.nn.tanh)
                self.copy_or_generate = tf.layers.dense(self.copy_or_generate, 1, activation=tf.nn.sigmoid)
                
                self.copy_probs = tf.matmul(copy, tf.one_hot(self.enc_seq_ids, self.wvocab_size))
                self.probs =  self.generate_probs * self.copy_or_generate + self.copy_probs * (1 - self.copy_or_generate)
                self.copy_pointer = tf.argmax(copy, axis=2)
            else:
                self.logits = self.output_layer(decoder_outputs)
                self.probs = tf.nn.softmax(self.logits, 2)
            self.beam_probs, self.beam_indices = tf.nn.top_k(self.probs, k=self.beam_width)
            self.sample_id = tf.argmax(self.probs, axis=2)
            self.top_k_value, self.top_k_indice = tf.nn.top_k(self.probs, k=2)

        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            with tf.variable_scope("loss") as scope:
                crossent = tf.reduce_sum(-tf.log(self.probs) * (
                    tf.one_hot(self.targets, self.wvocab_size) * 0.9 + 0.1 / self.wvocab_size),
                                         2)
                crosslabel = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.enc_edge_ids, logits=self.label_logits)
                crossdist = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.recon_edge_dists, logits=self.dist_logits)
                crossnode1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.enc_edge_links2, logits=self.link_logits1)
                crossnode2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.enc_edge_links1,
                                                                            logits=self.link_logits2)

                
                node_masks = tf.sequence_mask(self.enc_seq_lens, tf.shape(seq_outputs)[1], dtype=tf.float32)
                l2 = tf.zeros_like(self.enc_edge_ids, tf.int32)
                crosslabel_n1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=l2,
                                                                            logits=self.neg1_label_logits)
                crosslabel_n2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=l2,
                                                                               logits=self.neg2_label_logits)
                label_masks = tf.sequence_mask(self.enc_edge_lens, tf.shape(link1)[1], dtype=tf.float32)
                label_loss = 1.0 * tf.reduce_sum(crosslabel * label_masks)/self.batch_size \
                              + 0.1 * tf.reduce_sum(crosslabel_n1 * label_masks)/self.batch_size + 0.1 * tf.reduce_sum(crosslabel_n2 * label_masks)/self.batch_size
                node_loss = tf.reduce_sum(crossnode1 * label_masks + crossnode2 * label_masks)/self.batch_size
                dist_loss = tf.reduce_sum(crossdist * node_masks)/self.batch_size

                self.total_loss = tf.reduce_sum(crossent * self.weights)             
                self.loss = (tf.reduce_sum(crossent * self.weights)) / self.batch_size +  0.4 * label_loss + 0.1 * dist_loss


        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            with tf.variable_scope("train_op") as scope:
                self.global_step = tf.Variable(0, trainable=False)
                self.lr = noam_scheme(np.power(self.num_units, -0.5) * 0.35, self.global_step, 6000)
                optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.9, beta2=0.98, epsilon=1e-9)
                gradients, v = zip(*optimizer.compute_gradients(self.loss))
                self.train_op = optimizer.apply_gradients(zip(gradients, v), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables())

    def _single_cell(self, x=1):
        single_cell = tf.nn.rnn_cell.GRUCell(int(self.num_units/4))
        single_cell = tf.contrib.rnn.DropoutWrapper(single_cell,
                                                    input_keep_prob=0.75,
                                                    output_keep_prob=1)
        return single_cell

    def get_batch(self, data, no_random=False, id=0, if_pretrain=False, which=0):
        hparams = self.hparams
        enc_seq_ids = []
        enc_seq_lens = []
        enc_char_lens = []
        enc_char_ids = []
        enc_edge_ids = []
        enc_edge_lens = []
        enc_seq_masks = []
        enc_seq_masks_rev = []
        enc_edge_links1 = []
        enc_edge_links2 = []

        recon_edge_distids1 = []
        recon_edge_distids2 = []
        recon_edge_dists = []
        recon_edge_neglinks2 = []
        recon_edge_neglinks1 = []
        dec_masks = []
        dec_ids = []
        enc_depth = []
        dec_lens = []
        targets = []
        weights = []
        token_sum = 0

        for ii in range(int(hparams.batch_size)):
            if no_random:
                x, y, edges, z, d, _ = data[(id + ii) % len(data)]
            else:
                x, y, edges, z, d, _ = random.choice(data)
            if len(x) > self.max_src_length - 1:
                x = x[:self.max_src_length - 1]
            if len(y) > self.max_tgt_length - 2:
                y = y[:self.max_tgt_length - 2]
            if len(z) > self.max_src_length - 1:
                z = z[:self.max_src_length - 1]
            if len(d) > self.max_src_length - 1:
                d = d[:self.max_src_length - 1]

            lx = len(x)
            ly = len(y)
            depth = []
            for w in d:
                if len(w) - 1 <= 6:
                    depth.append(len(w) - 1)
                else:
                    depth.append(7)
            enc_seq_ids.append(x + [PAD_ID] * (self.max_src_length - lx))
            enc_depth.append(depth + [0] * (self.max_src_length - lx))
            dec_ids.append([GO_ID] + y + [PAD_ID] * (self.max_tgt_length - ly - 1))
            targets.append(y + [EOS_ID] + [PAD_ID] * (self.max_tgt_length - ly - 1))
            weights.append([1.0] * (ly + 1) + [0.0] * (self.max_tgt_length - ly - 1))
            enc_seq_lens.append(lx)
            dec_lens.append(ly + 1)
            tmp = [0] * self.max_tgt_length
            masks = []
            char_len = []
            enc_char_id = []
            for chars in d:
                if len(chars) >= 20:
                    c = chars[:20]
                else:
                    c = chars
                char_len.append(len(c))
                enc_char_id.append(c + [0] * (20 - len(c)))
            for _ in range(self.max_src_length - lx):
                c = [0]
                char_len.append(len(c))
                enc_char_id.append(c + [0] * (20 - len(c)))
            enc_char_lens.append(char_len)
            enc_char_ids.append(enc_char_id)
            for i in range(0, self.max_tgt_length):
                tmp[i] = 1
                masks.append(tmp.copy())
            
            dec_masks.append(masks)
            enc_edge_id = []
            enc_edge_link1 = []
            enc_edge_link2 = []
            enc_seq_mask = [[0] * self.max_src_length for _ in range(self.max_src_length)]
            enc_seq_mask_rev = [[0] * self.max_src_length for _ in range(self.max_src_length)]

            recon_edge_distid1 = []
            recon_edge_distid2 = []
            recon_edge_dist = []
            recon_edge_neglink2 = []
            recon_edge_neglink1 = []

            
            tmp = [[-1] * self.max_src_length for _ in range(self.max_src_length)]
            tmp2 = [[0] * self.max_src_length for _ in range(self.max_src_length)]

            i = 0
            for l in edges:
                edge, id1, id2 = l
                if id1 >= self.max_src_length - 1 or id2 >= self.max_src_length - 1:
                    continue
                enc_edge_id.append(edge)
                enc_edge_link1.append(id1)
                enc_edge_link2.append(id2)
                x = random.randint(0,lx-1)
                if x == id2:
                    x = (x+1) % lx
                recon_edge_neglink2.append(x)
                x = random.randint(0, lx - 1)
                if x == id1:
                    x = (x + 1) % lx
                recon_edge_neglink1.append(x)
                enc_seq_mask[id1][i] = 1
                enc_seq_mask_rev[id2][i] = 1
                tmp[id1][id2] = 1
                tmp[id2][id1] = 1
                tmp2[id1][id2] = 1
                tmp2[id2][id1] = -1
                
                i += 1
                if i >= self.max_src_length:
                    break
            le = len(enc_edge_id)

            t = 0
            for k in range(lx):
                tmp[k][k] = 0
                for i in range(lx):
                    if tmp[i][k] == -1:
                        continue
                    for j in range(lx):
                        if tmp[k][j] == -1:
                            continue
                        if i == j:
                            continue
                       
                        if tmp[i][j] == -1 or tmp[i][k] + tmp[k][j] < tmp[i][j]:
                            tmp[i][j] = tmp[i][k] + tmp[k][j]

                        t += 1
                        if t >= self.max_src_length * 1:
                            break
                    if t >= self.max_src_length * 1:
                        break
                if t >= self.max_src_length * 1:
                    break






            for i in range(lx):
                x = random.randint(0, lx - 1)
                y = random.randint(0, lx - 1)
                recon_edge_distid1.append(x)
                recon_edge_distid2.append(y)
                if tmp[x][y] == -1 or tmp[x][y] >=13:
                    recon_edge_dist.append(13)
                else:
                    recon_edge_dist.append(tmp[x][y])
            

            recon_edge_distids1.append(recon_edge_distid1 + [PAD_ID] * (self.max_src_length * 1 - lx))
            recon_edge_distids2.append(recon_edge_distid2 + [PAD_ID] * (self.max_src_length * 1 - lx))
            recon_edge_dists.append(recon_edge_dist + [PAD_ID] * (self.max_src_length * 1 - lx))
            recon_edge_neglinks2.append(recon_edge_neglink2 + [PAD_ID] * (self.max_src_length * 1 - le))
            recon_edge_neglinks1.append(recon_edge_neglink1 + [PAD_ID] * (self.max_src_length * 1 - le))
            enc_edge_lens.append(le)
            enc_edge_ids.append(enc_edge_id + [PAD_ID] * (self.max_src_length - le))
            enc_edge_links1.append(enc_edge_link1 + [PAD_ID] * (self.max_src_length - le))
            enc_edge_links2.append(enc_edge_link2 + [PAD_ID] * (self.max_src_length - le))
            enc_seq_masks.append(enc_seq_mask)
            enc_seq_masks_rev.append(enc_seq_mask_rev)
            token_sum += ly + 1
            
        return enc_seq_ids, enc_seq_lens, enc_seq_masks, enc_seq_masks_rev, enc_edge_ids, enc_edge_links1, enc_edge_links2, enc_edge_lens, \
               recon_edge_dists, recon_edge_neglinks2, recon_edge_neglinks1,  recon_edge_distids1, recon_edge_distids2,\
               enc_char_ids, enc_char_lens, enc_depth, dec_ids, dec_lens, dec_masks, targets, weights

    def train_step(self, sess, data, no_random=False, id=0, if_pretrain=False):

        enc_seq_ids, enc_seq_lens, enc_seq_masks, enc_seq_masks_rev, enc_edge_ids, enc_edge_links1, enc_edge_links2, enc_edge_lens, \
         recon_edge_dists, recon_edge_neglinks2, recon_edge_neglinks1, recon_edge_distids1, recon_edge_distids2, \
        enc_char_ids, enc_char_lens, enc_depth, dec_ids, dec_lens, dec_masks, targets, weights = \
            self.get_batch(data, no_random=no_random, id=id, if_pretrain=if_pretrain)
        feed = {
            self.enc_seq_ids: enc_seq_ids,
            self.enc_seq_lens: enc_seq_lens,
            self.enc_seq_masks: enc_seq_masks,
            self.enc_seq_masks_rev: enc_seq_masks_rev,
            self.enc_edge_ids: enc_edge_ids,
            self.enc_depth: enc_depth,
            self.enc_edge_lens: enc_edge_lens,
            self.enc_edge_links1: enc_edge_links1,
            self.enc_edge_links2: enc_edge_links2,
            self.dec_masks: dec_masks,
            self.weights: weights,
            self.targets: targets,
            self.enc_char_ids: enc_char_ids,
            self.enc_char_lens: enc_char_lens,
            self.dec_ids: dec_ids,
            self.dec_lens: dec_lens,
            self.recon_edge_distids1: recon_edge_distids1,
            self.recon_edge_distids2: recon_edge_distids2,
            self.recon_edge_dists: recon_edge_dists,
            self.recon_edge_neglinks2: recon_edge_neglinks2,
            self.recon_edge_neglinks1: recon_edge_neglinks1,
        }

        word_nums = sum(sum(weight) for weight in weights) 
        loss, global_step, _, total_loss = sess.run([self.loss, self.global_step, self.train_op, self.total_loss],
                                                    feed_dict=feed)
    

        return total_loss, global_step, word_nums

    def eval_step(self, sess, data, no_random=False, id=0):
        enc_seq_ids, enc_seq_lens, enc_seq_masks, enc_seq_masks_rev, enc_edge_ids, enc_edge_links1, enc_edge_links2, enc_edge_lens, \
        recon_edge_dists, recon_edge_neglinks2, recon_edge_neglinks1, recon_edge_distids1, recon_edge_distids2,\
        enc_char_ids, enc_char_lens, enc_depth, dec_ids, dec_lens, dec_masks, targets, weights = self.get_batch(
            data, no_random=no_random, id=id)
        feed = {
            self.enc_seq_ids: enc_seq_ids,
            self.enc_seq_lens: enc_seq_lens,
            self.enc_seq_masks: enc_seq_masks,
            self.enc_seq_masks_rev: enc_seq_masks_rev,
            self.enc_edge_ids: enc_edge_ids,
            self.enc_edge_lens: enc_edge_lens,
            self.enc_depth: enc_depth,
            self.enc_edge_links1: enc_edge_links1,
            self.enc_edge_links2: enc_edge_links2,
            self.enc_char_ids: enc_char_ids,
            self.enc_char_lens: enc_char_lens,
            self.recon_edge_distids1:recon_edge_distids1,
            self.recon_edge_distids2:recon_edge_distids2,
            self.recon_edge_dists:recon_edge_dists,
            self.recon_edge_neglinks2:recon_edge_neglinks2,
            self.recon_edge_neglinks1:recon_edge_neglinks1,
            self.dec_masks: dec_masks,
            self.weights: weights,
            self.targets: targets,
            self.dec_ids: dec_ids,
            self.dec_lens: dec_lens,
        }
        loss = sess.run(self.total_loss, feed_dict=feed)
        word_nums = sum(sum(weight) for weight in weights)
        return loss, word_nums

    def infer_step(self, sess, data, no_random=False, id=0, postag=None, which=0):
        enc_seq_ids, enc_seq_lens, enc_seq_masks, enc_seq_masks_rev, enc_edge_ids, enc_edge_links1, enc_edge_links2, enc_edge_lens, \
        recon_edge_dists, recon_edge_neglinks2, recon_edge_neglinks1, recon_edge_distids1, recon_edge_distids2,\
        enc_char_ids, enc_char_lens, enc_depth, dec_ids, dec_lens, dec_masks, targets, weights = self.get_batch(
            data, no_random=no_random, id=id)
        start_pos = []
        given = []
        ans = []
        predict = []
        for i in range(self.hparams.batch_size):
            given.append(enc_seq_ids[i][:enc_seq_lens[i]])
            ans.append(targets[i].copy())
        dict = [0] * self.hparams.batch_size
        ct = 0
        for i in range(self.max_tgt_length - 2):
            feed = {
                self.enc_seq_ids: enc_seq_ids,
                self.enc_seq_lens: enc_seq_lens,
                self.enc_seq_masks: enc_seq_masks,
                self.enc_depth: enc_depth,
                self.enc_seq_masks_rev: enc_seq_masks_rev,
                self.enc_edge_ids: enc_edge_ids,
                self.enc_edge_lens: enc_edge_lens,
                self.enc_edge_links1: enc_edge_links1,
                self.enc_edge_links2: enc_edge_links2,
                self.recon_edge_distids1: recon_edge_distids1,
                self.recon_edge_distids2: recon_edge_distids2,
                self.recon_edge_dists: recon_edge_dists,
                self.recon_edge_neglinks2: recon_edge_neglinks2,
                self.recon_edge_neglinks1: recon_edge_neglinks1,
                self.enc_char_ids: enc_char_ids,
                self.enc_char_lens: enc_char_lens,
                self.dec_masks: dec_masks,
                self.dec_ids: dec_ids,
                self.dec_lens: dec_lens,
            }
            sample_id, top_value, top_indice = sess.run([self.sample_id, self.top_k_value, self.top_k_indice],
                                                        feed_dict=feed)
            for batch in range(self.hparams.batch_size):
                if sample_id[batch][i] != UNK_ID and sample_id[batch][i] != dec_ids[batch][i]:
                    dec_ids[batch][i + 1] = sample_id[batch][i]
                else:
                    dec_ids[batch][i + 1] = top_indice[batch][i][1]
                if dec_ids[batch][i + 1] == EOS_ID and dict[batch] == 0:
                    dict[batch] = 1
                    ct += 1
            if ct == self.hparams.batch_size:
                break

        return given, dec_ids

    def min(self, a, b):
        if a < b:
            return a
        else:
            return b

    def infer_step_beam(self, sess, data, no_random=True, id=0):
        enc_seq_ids, enc_seq_lens, enc_seq_masks, enc_seq_masks_rev, enc_edge_ids, enc_edge_links1, enc_edge_links2, enc_edge_lens, \
        recon_edge_dists, recon_edge_neglinks2, recon_edge_neglinks1, recon_edge_distids1, recon_edge_distids2,\
        enc_char_ids, enc_char_lens, enc_depth, dec_ids, dec_lens, dec_masks, targets, weights = self.get_batch(
            data, no_random=no_random, id=id)

        given = []
        ans = []
        for i in range(self.hparams.batch_size):
            given.append(enc_seq_ids[i][:enc_seq_lens[i]])
            ans.append(targets[i].copy())
        dict = [0] * self.hparams.batch_size
        
        ct = 0


        feed = {
            self.enc_seq_ids: enc_seq_ids,
            self.enc_seq_lens: enc_seq_lens,
            self.enc_depth: enc_depth,
            self.enc_seq_masks: enc_seq_masks,
            self.enc_seq_masks_rev: enc_seq_masks_rev,
            self.enc_edge_ids: enc_edge_ids,
            self.enc_edge_lens: enc_edge_lens,
            self.enc_edge_links1: enc_edge_links1,
            self.enc_edge_links2: enc_edge_links2,
            self.enc_char_ids: enc_char_ids,
            self.enc_char_lens: enc_char_lens,
            self.dec_masks: dec_masks,
            self.dec_ids: dec_ids,
            self.dec_lens: dec_lens,
            self.recon_edge_distids1: recon_edge_distids1,
            self.recon_edge_distids2: recon_edge_distids2,
            self.recon_edge_dists: recon_edge_dists,
            self.recon_edge_neglinks2: recon_edge_neglinks2,
            self.recon_edge_neglinks1: recon_edge_neglinks1,
        }

        ans = [0] * self.hparams.batch_size
        probs, indices = sess.run([self.beam_probs, self.beam_indices], feed_dict=feed)
        beam_inputs = np.array([[[0] * self.max_tgt_length] * self.hparams.batch_size] * self.beam_width)
        beam_probs = np.array([[-1000000000.0] * self.hparams.batch_size] * self.beam_width)
        ref_y = []
        for j in range(self.hparams.batch_size):
            ref_y.append(int(1.2 * enc_seq_lens[j]))
            for k in range(self.beam_width):
                beam_inputs[k][j][1] = indices[j][0][k]
                beam_probs[k][j] = math.log(probs[j][0][k])

        for i in range(1, self.max_tgt_length - 2):
            all_inputs = []
            all_probs = []
            for j in range(0, self.batch_size):
                dec_lens[j] = i + 2
            for j in range(0, self.beam_width):
                feed = {
                    self.enc_seq_ids: enc_seq_ids,
                    self.enc_seq_lens: enc_seq_lens,
                    self.enc_seq_masks: enc_seq_masks,
                    self.enc_depth: enc_depth,
                    self.enc_seq_masks_rev: enc_seq_masks_rev,
                    self.enc_edge_ids: enc_edge_ids,
                    self.enc_edge_lens: enc_edge_lens,
                    self.enc_edge_links1: enc_edge_links1,
                    self.enc_edge_links2: enc_edge_links2,
                    self.enc_char_ids: enc_char_ids,
                    self.enc_char_lens: enc_char_lens,
                    self.dec_masks: dec_masks,
                    self.dec_ids: beam_inputs[j],
                    self.dec_lens: dec_lens,
                    self.recon_edge_distids1: recon_edge_distids1,
                    self.recon_edge_distids2: recon_edge_distids2,
                    self.recon_edge_dists: recon_edge_dists,
                    self.recon_edge_neglinks2: recon_edge_neglinks2,
                    self.recon_edge_neglinks1: recon_edge_neglinks1,
                }

                probs, indices = sess.run([self.beam_probs, self.beam_indices], feed_dict=feed)
                tmp_inputs = []
                tmp_probs = []
                for k in range(self.beam_width):
                    x = beam_inputs[j].copy()
                    y = beam_probs[j].copy()
                    tmp_inputs.append(x)
                    tmp_probs.append(y)


                for k in range(self.hparams.batch_size):
                    check = 1
                    x = i + 1
                    for p in range(1, i + 1):
                        if beam_inputs[j][k][x] == EOS_ID:
                            check = 0
                            break
                    for l in range(0, self.beam_width):
                        tmp_inputs[l][k][i + 1] = indices[k][i][l]

                        if check == 1:
                            tmp_probs[l][k] += math.log(probs[k][i][l])    
                        elif check == 0:
                            check = -1
                            tmp_probs[l][k] += 0 
                        else:
                            tmp_probs[l][k] -= 10000
                        

                all_inputs.extend(tmp_inputs)
                all_probs.extend(tmp_probs)
               
            all_inputs = np.transpose(np.array(all_inputs), [1, 0, 2])
            all_probs = np.transpose(np.array(all_probs), [1, 0])

            for batch in range(self.hparams.batch_size):
                topk = np.argsort(-all_probs[batch])
                for j in range(self.beam_width):
                    beam_probs[j][batch] = all_probs[batch][topk[j]]
                    beam_inputs[j][batch] = all_inputs[batch][topk[j]]
                if EOS_ID in beam_inputs[0][batch] and EOS_ID in beam_inputs[1][batch] and EOS_ID in beam_inputs[2][batch] and EOS_ID in beam_inputs[3][batch] and EOS_ID in beam_inputs[4][batch] and dict[batch] == 0:
                    dict[batch] = 1
                    ct += 1
            if ct == self.hparams.batch_size:
                break

        sample_id = []
        beam_inputs = beam_inputs.tolist()
        sample_lens = []
        for i in range(0, self.batch_size):
            sample_lens.append(self.max_tgt_length)
            for j in range(0, self.beam_width):
                beam_inputs[j][i].append(EOS_ID)
                x = beam_inputs[j][i].index(EOS_ID)
                beam_inputs[j][i] = beam_inputs[j][i][:self.max_tgt_length]
                beam_probs[j][i] = beam_probs[j][i] / math.pow((5 + x) / 6, 0.0)

        all_probs = np.transpose(np.array(beam_probs), [1, 0])
        for batch in range(0, self.batch_size):
            topk = np.argsort(-all_probs[batch])
            sample_id.append(beam_inputs[topk[0]][batch][:self.max_tgt_length])

        feed = {
            self.enc_seq_ids: enc_seq_ids,
            self.enc_seq_lens: enc_seq_lens,
            self.enc_seq_masks: enc_seq_masks,
            self.enc_depth: enc_depth,
            self.enc_seq_masks_rev: enc_seq_masks_rev,
            self.enc_edge_ids: enc_edge_ids,
            self.enc_edge_lens: enc_edge_lens,
            self.enc_edge_links1: enc_edge_links1,
            self.enc_edge_links2: enc_edge_links2,
            self.enc_char_ids: enc_char_ids,
            self.enc_char_lens: enc_char_lens,
            self.dec_masks: dec_masks,
            self.dec_ids: sample_id,
            self.dec_lens: sample_lens,
            self.recon_edge_distids1: recon_edge_distids1,
            self.recon_edge_distids2: recon_edge_distids2,
            self.recon_edge_dists: recon_edge_dists,
            self.recon_edge_neglinks2: recon_edge_neglinks2,
            self.recon_edge_neglinks1: recon_edge_neglinks1,
        }

        align = sess.run(self.copy_pointer, feed_dict=feed)
        return given, sample_id, align

    def lr_decay(self, sess):
        return sess.run(self.learning_rate_decay_op)

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)