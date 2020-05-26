#!/usr/bin/python3
from __future__ import print_function

"""
Copyright (c) 2018, University of North Carolina at Charlotte All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors: Reza Baharani - Transformative Computer Systems Architecture Research (TeCSAR) at UNC Charlotte
"""


"""
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
Inspired by
    https://github.com/aymericdamien/TensorFlow-Examples/
    and
    http://mourafiq.com/2016/05/15/predicting-sequences-using-rnn-in-tensorflow.html
    and
    https://github.com/sunsided/tensorflow-lstm-sin
"""

# noinspection PyUnresolvedReferences
import os
import matplotlib
import argparse
import math
from utility.generate_sample import generate_sample
import scipy.io as matloader
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


import matplotlib.pyplot as plt
matplotlib.use("Agg")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


_DISCRIPTION = '''
The file dR11Devs has following devices:\n
idx    Device\n
idx    Device\n
00 -> 'Dev#8'\n
01 -> 'Dev#9'\n
02 -> 'Dev#11'\n
03 -> 'Dev#12'\n
04 -> 'Dev#14'\n
05 -> 'Dev#24'\n
06 -> 'Dev#29'\n
07 -> 'Dev#32'\n
08 -> 'Dev#35'\n
09 -> 'Dev#36'\n
10 -> 'Dev#38'\n
'''

parser = argparse.ArgumentParser(
    description='dR Transistor Degradation predicion Based on Stacked LSTM Approch.')
parser.add_argument('--test-dev', type=int, default=14,
                    help="Device test ID. {}".format(_DISCRIPTION))
parser.add_argument('--rul-time', type=int, nargs='*',
                    help="Calculate the RUL at given time.")
parser.add_argument('--threshold', type=float, default=0.05,
                    help="Calculate the RUL at given time.")

args = parser.parse_args()

if not(-1 < args.test_dev < 10):
    print(
        "test-dev should be a number in [0,9]. Please run thte program with --help for more information.")
    raise ValueError


# Parameters
data_file = "./utility/dR11Devs.mat"

# Network Parameters
n_input = 1  # Delta{R}
n_steps = 20  # time steps
n_hidden = 32  # Num of features
n_outputs = 104  # output is a series of Delta{R}+
n_layers = 4  # number of stacked LSTM layers
save_res_as_file = True
fig_output_dir = './pred_plots/'
mat = matloader.loadmat(data_file)

test_device = [args.test_dev]
_, _, _, _, _, dev_name = generate_sample(
    filename=data_file, batch_size=1, samples=n_steps, predict=n_outputs, test=True, test_set=test_device)

'''
    Computation graph
'''

with tf.compat.v1.name_scope("INPUTs"):
    # tf Graph input
    lr = tf.compat.v1.placeholder(tf.float32, [])
    x = tf.compat.v1.placeholder(tf.float32, [None, n_steps, n_input])
    y = tf.compat.v1.placeholder(tf.float32, [None, n_outputs])

# Define weights
weights = {
    'out': tf.Variable(tf.random.truncated_normal([n_hidden, n_outputs], stddev=1.0))
}

biases = {
    'out': tf.Variable(tf.random.truncated_normal([n_outputs], stddev=0.1))
}

# Define the GRU cells
# with tf.name_scope("GRU_CELL"):
#    gru_cells = [rnn.GRUCell(n_hidden) for _ in range(n_layers)]
# with tf.name_scope("GRU_NETWORK"):
#    stacked_lstm = rnn.MultiRNNCell(gru_cells)

# Define the LSTM cells
with tf.compat.v1.name_scope("LSTM_CELL"):
    lstm_cells = [tf.compat.v1.nn.rnn_cell.LSTMCell(
        n_hidden, forget_bias=1.) for _ in range(n_layers)]
# with tf.name_scope("LSTM"):
    stacked_lstm = tf.compat.v1.nn.rnn_cell.MultiRNNCell(lstm_cells)

with tf.compat.v1.name_scope("OUTPUTs"):
    outputs, states = tf.compat.v1.nn.dynamic_rnn(
        stacked_lstm, inputs=x, dtype=tf.float32, time_major=False)
    h = tf.transpose(a=outputs, perm=[1, 0, 2])
    pred = tf.nn.bias_add(tf.matmul(h[-1], weights['out']), biases['out'])


'''
We don't need this part for inference
'''
# Define loss (Euclidean distance) and optimizer
#individual_losses = tf.reduce_sum(tf.squared_difference(pred, y), reduction_indices=1)
# with tf.name_scope("Loss"):
#    loss = tf.reduce_mean(individual_losses)
#    tf.summary.scalar("loss", loss)

#optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# Initializing the variables
init = tf.compat.v1.global_variables_initializer()

merged = tf.compat.v1.summary.merge_all()

load_model = True
model_file_name = './inference_models/model_' + dev_name + '.ckpt'

saver = tf.compat.v1.train.Saver()
TX2_Board_Power = False

config = tf.compat.v1.ConfigProto(
    device_count={'GPU': 0}
)


def calc_error(output, target, index):
    return abs((output[index] - target[index])) / target[index]


def find_idx(arr):
    if not arr.any():
        return -1
    return np.argmax(arr)


if __name__ == "__main__":

    with tf.compat.v1.Session(config=config) as sess:

        if load_model:
            # Restore variables from disk.
            saver.restore(sess, model_file_name)
            print('Model restored.')
        else:
            sess.run(init)

        from datetime import datetime
        _, _, _, _, l, dev_name = generate_sample(
            filename=data_file, batch_size=1, samples=n_steps, predict=n_outputs, test=True, test_set=test_device)
        total_dev_len = l[0]
        how_many_seg = int(total_dev_len / (n_steps + n_outputs))

        avg_Time = .0

        if TX2_Board_Power:
            import socket
            # Create a TCP/IP socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            # Connect the socket to the port where the server is listening
            server_address = ('localhost', 8080)
            print('connecting to {} port {}'.format(*server_address))
            sock.connect(server_address)

            sent_before = False
            message = b'START\n'
            print('sending {!r}'.format(message))
            sock.sendall(message)

        # Has only network prediction withou input seq.
        out = np.array([])
        tr = np.array([])

        pred_lst = np.array([])
        out_lst = np.array([])

        for i in range(how_many_seg):
            _, y, next_t, expected_y, _, _ = generate_sample(filename=data_file,
                                                             batch_size=1, samples=n_steps, predict=n_outputs,
                                                             start_from=i * (n_steps + n_outputs), test=True, test_set=test_device)
            test_input = y.reshape((1, n_steps, n_input))
            tstart = datetime.now()
            prediction = sess.run(pred, feed_dict={x: test_input})
            tend = datetime.now()

            if TX2_Board_Power:
                if not sent_before:
                    message = b'STOP\n'
                    print('sending {!r}'.format(message))
                    sock.sendall(message)
                    sent_before = True

            # remove the batch size dimensions
            delta = tend - tstart

            pred_lst = np.hstack((pred_lst, y[0]))  # Input Seq
            pred_lst = np.hstack((pred_lst, prediction[0]))  # Prediction

            out_lst = np.hstack((out_lst, y[0]))
            out_lst = np.hstack((out_lst, expected_y[0]))

            out = np.hstack((out, prediction[0]))
            tr = np.hstack((tr, expected_y[0]))

            avg_Time += (delta.total_seconds())*1000
            print('Next loop: {:3.2f}'.format(
                int(i*100/(how_many_seg-1))), end='\r')

        pred_lst_5p = np.array([])
        out_lst_5p = np.array([])

        # Calculating hot zone before threshold.

        final_gold = mat['vals'][0, args.test_dev][0]
        thr_idx = find_idx(final_gold >= args.threshold)
        if thr_idx == -1:
            thr_idx = total_dev_len-1
            print("{} cann't be defined as threshould value. I changed it to the {}.".format(
                args.threshold, out_lst[thr_idx]))

        _, y_p, _, expected_y_p, _, _ = generate_sample(filename=data_file,
                                                        batch_size=1, samples=n_steps, predict=n_outputs,
                                                        start_from=thr_idx - (n_steps + n_outputs), test=True, test_set=test_device)

        test_input_5p = y_p.reshape((1, n_steps, n_input))
        tstart = datetime.now()
        prediction_5p = sess.run(pred, feed_dict={x: test_input_5p})

        pred_lst_5p = np.hstack((pred_lst_5p, y_p[0]))  # Input Seq
        pred_lst_5p = np.hstack((pred_lst_5p, prediction_5p[0]))  # Prediction
        out_lst_5p = np.hstack((out_lst_5p, y_p[0]))
        out_lst_5p = np.hstack((out_lst_5p, expected_y_p[0]))

        error = calc_error(prediction_5p[0], expected_y_p[0], -1)
        error *= 100
        print('target = {:0.6f} | prediction = {:0.6f} | error: {:3.4f}%'.format(
            expected_y_p[0][-1], prediction_5p[0][-1], error))

        if save_res_as_file:
            #pred_nump = np.array(pred_lst)
            np.savetxt('./prediction_output/res_' +
                       dev_name + '.txt', pred_lst, fmt="%f", newline='\r\n')

            plt.figure(figsize=(8.5, 2.85))
            plt.plot(range(len(pred_lst)), pred_lst, 'b',
                     label='{} - Predicted'.format(dev_name.replace('_', '#')))
            plt.plot(range(len(out_lst)), out_lst, 'r', alpha=0.6,
                     label='{} - Measured'.format(dev_name.replace('_', '#')))

            plt.legend(loc='upper left')
            plt.xlabel('Num. of Samples')
            plt.ylabel('ΔR')
            plt.savefig(
                '{}/device_{}_full.png'.format(fig_output_dir, dev_name))

            plt.clf()
            plt.cla()
            plt.close()

            plt.figure(figsize=(8.5, 2.85))
            plt.plot(range(len(pred_lst_5p)), pred_lst_5p, 'b',
                     label='{} - Predicted'.format(dev_name.replace('_', '#')))
            plt.plot(range(len(out_lst_5p)), out_lst_5p, 'r', alpha=0.6,
                     label='{} - Measured'.format(dev_name.replace('_', '#')))

            plt.legend(loc='upper left')
            plt.xlabel('Num. of Samples')
            plt.ylabel('ΔR')
            plt.savefig(
                '{}/device_{}_tail.png'.format(fig_output_dir, dev_name))

        mse = ((out - tr)**2).mean(axis=0)
        print("Log(MSE) of {} is {:2.4f}".format(dev_name, math.log10(mse)))

        mse_5p = ((expected_y_p[0] - prediction_5p[0])**2).mean(axis=0)
        print("Log(MSE) of {} at hot zone is {:2.4f}".format(
            dev_name, math.log10(mse_5p)))

        print('Avg elapsed time for predicting one minute:  {:2.4} mS'.format(
            avg_Time/how_many_seg))

        if len(args.rul_time) > 0:
            print('+'*100)
            samplePerMin = mat['SamplePerMinuts'][0, args.test_dev][0]
            degAt = int(thr_idx/samplePerMin[0])
            print("{} is degregaded at {} minutes.".format(dev_name, degAt))
            print('-'*100)
            for rul_time in args.rul_time:
                print("Calculating the RUL at {} considering threshols={}.".format(
                    rul_time, args.threshold))
                if rul_time > degAt:
                    raise ValueError(
                        "Given time for RUL is bigger than {}.".format(degAt))
                rul_time_idx = int(rul_time * samplePerMin[0])
                _, y_p_rul, _, expected_y_p_rul, _, _ = generate_sample(filename=data_file,
                                                                        batch_size=1, samples=n_steps, predict=n_outputs,
                                                                        start_from=rul_time_idx - (n_steps + n_outputs), test=True, test_set=test_device)

                test_input_rul = y_p_rul.reshape((1, n_steps, n_input))
                prediction_rul = sess.run(pred, feed_dict={x: test_input_rul})
                RUL = 100 * \
                    abs(final_gold[rul_time_idx] -
                        final_gold[-1])/final_gold[-1]
                RUL_hat = 100 * \
                    abs(prediction_5p[0][-1] - expected_y_p_rul[0]
                        [-1])/prediction_5p[0][-1]
                print("RUL at {} is {:3.1f}".format(rul_time, RUL))
                print("Estimated RUL at {} is {:3.1f}".format(
                    rul_time, RUL_hat))
                RA = 100*(1-(abs(RUL-RUL_hat)/RUL))
                Err = RUL - RUL_hat
                print("Error at {} is {:3.1f}".format(rul_time, Err))
                print("RA at {} is {:3.1f}".format(rul_time, RA))
                print('-'*100)

        if TX2_Board_Power:
            print('closing socket')
            sock.close()
