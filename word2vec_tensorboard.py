import argparse
import os
import pickle
import traceback

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


def turn_off_tensorflow_logging():
    import os
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ignore tensorflow warnings
    tf.logging.set_verbosity(tf.logging.ERROR)  # ignore tensorflow info (GPU 할당 정보 확인)


def word2vec_tensorboard(name, data_dir, tensorboard_dir, top_n=10000):
    turn_off_tensorflow_logging()
    try:
        if not os.path.exists(tensorboard_dir):
            os.mkdir(tensorboard_dir)
        for filename in os.listdir(tensorboard_dir):
            os.remove(os.path.join(tensorboard_dir, filename))  # remove old tensorboard files

        config = projector.ProjectorConfig()

        name = name.replace('+', '')

        idx2word = pickle.load(open(os.path.join(data_dir, 'idx2word.dat'), 'rb'))
        # word2idx = pickle.load(open('data/word2idx.dat', 'rb'))
        idx2vec = pickle.load(open(os.path.join(data_dir, 'idx2vec.dat'), 'rb'))
        wc = pickle.load(open(os.path.join(data_dir, 'wc.dat'), 'rb'))
        total = sum(wc.values())

        # print('idx2word:', idx2word[:10])
        # print('idx2vec:', idx2vec[1])
        # print('wc:', list(wc.items())[:10])
        print('total count:', total)

        idx2vec, idx2word = idx2vec[:top_n], idx2word[:top_n]

        embedding_var = tf.Variable(idx2vec, name=name)
        # print(data)
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        embedding.metadata_path = os.path.join(tensorboard_dir, f'{name}.tsv')

        print('')
        print(f'embedding_var.name: {embedding_var.name} shape: {embedding_var.shape}')
        print(f'embedding.metadata_path: {embedding.metadata_path}')
        with open(embedding.metadata_path, 'wt') as out_f:
            out_f.write('spell\tfreq\n')
            for spell in idx2word:
                out_f.write(f'{spell}\t{wc.get(spell, 0)/total}\n')

        summary_writer = tf.summary.FileWriter(tensorboard_dir)
        projector.visualize_embeddings(summary_writer, config)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(var_list=[embedding_var])
            checkpoint_file = os.path.join(tensorboard_dir, f'{name}.ckpt')
            saver.save(sess, checkpoint_file, global_step=None)
            print(f'checkpoint_file: {checkpoint_file}')

        # absolute path -> relative path
        for filename in ['checkpoint', 'projector_config.pbtxt']:
            filepath = os.path.join(tensorboard_dir, filename)

            lines = []
            with open(filepath, 'rt') as f:
                for line in f.readlines():
                    lines.append(line.replace(tensorboard_dir, '.'))
            os.remove(filepath)
            with open(filepath, 'wt') as f:
                for line in lines:
                    f.write(line)
    except:
        traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='sample.ko.wikipedia', type=str, help="embedding name in tensorboard projector")
    parser.add_argument('--data_dir', default=os.path.join(os.getenv('HOME'), 'workspace/word2vec4kor/data'), type=str, help="data directory path")
    parser.add_argument('--tensorboard_dir', default=os.path.join(os.getenv('HOME'), 'tensorboard_log/'), type=str, help="tensorboard directory path")
    parser.add_argument('--top_n', default=10000, type=int, help='max number of vocaburary')
    args = parser.parse_args()

    word2vec_tensorboard(name=args.name, data_dir=args.data_dir, tensorboard_dir=args.tensorboard_dir, top_n=args.top_n)
