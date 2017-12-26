import logging
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sn
import matplotlib.pyplot as plt


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics


# Hyperparameters
L1_HIDDEN = 375
L2_HIDDEN = 200
L3_HIDDEN = 150
L4_HIDDEN = 75
L5_HIDDEN = 50
DROPOUT_RATIO = 0.2

# TODO: Datasets require different preparation, should isolate and give option
TRAINING_FILE = 'KDDTrain+.txt'
TEST_FILE = 'KDDTest+.txt'
# TEST_FILE = 'kddcup.data.corrected'
# TRAINING_FILE = 'kdd/kddcup.data10'
# TEST_FILE = 'kdd/corrected'

"""
    Dictionary of attack types
"""
ATTACK_DICT = dict()
ATTACK_DICT['normal'] = 'normal'

ATTACK_DICT['apache2'] = 'dos'
ATTACK_DICT['back'] = 'dos'
ATTACK_DICT['land'] = 'dos'
ATTACK_DICT['mailbomb'] = 'dos'
ATTACK_DICT['neptune'] = 'dos'
ATTACK_DICT['pod'] = 'dos'
ATTACK_DICT['processtable'] = 'dos'
ATTACK_DICT['smurf'] = 'dos'
ATTACK_DICT['teardrop'] = 'dos'
ATTACK_DICT['udpstorm'] = 'dos'

ATTACK_DICT['buffer_overflow'] = 'u2r'
ATTACK_DICT['loadmodule'] = 'u2r'
ATTACK_DICT['perl'] = 'u2r'
ATTACK_DICT['rootkit'] = 'u2r'
ATTACK_DICT['ps'] = 'u2r'
ATTACK_DICT['sqlattack'] = 'u2r'
ATTACK_DICT['xterm'] = 'u2r'
ATTACK_DICT['worm'] = 'u2r'
ATTACK_DICT['snmpguess'] = 'u2r'


ATTACK_DICT['httptunnel'] = 'r2l'
ATTACK_DICT['spy'] = 'r2l'
ATTACK_DICT['warezclient'] = 'r2l'
ATTACK_DICT['ftp_write'] = 'r2l'
ATTACK_DICT['guess_passwd'] = 'r2l'
ATTACK_DICT['imap'] = 'r2l'
ATTACK_DICT['multihop'] = 'r2l'
ATTACK_DICT['named'] = 'r2l'
ATTACK_DICT['phf'] = 'r2l'
ATTACK_DICT['warezmaster'] = 'r2l'
ATTACK_DICT['sendmail'] = 'r2l'
ATTACK_DICT['snmpgetattack'] = 'r2l'
ATTACK_DICT['xlock'] = 'r2l'
ATTACK_DICT['xsnoop'] = 'r2l'

ATTACK_DICT['ipsweep'] = 'probe'
ATTACK_DICT['mscan'] = 'probe'
ATTACK_DICT['nmap'] = 'probe'
ATTACK_DICT['portsweep'] = 'probe'
ATTACK_DICT['saint'] = 'probe'
ATTACK_DICT['satan'] = 'probe'

"""
    Dataset columns
"""
COLUMNS = [
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'outcome',
    'difficulty']

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def encode_one_hot(df, column):
    """
        One hot encoding for DataFrame categorical column
    """
    dummies = pd.get_dummies(df[column])

    # noinspection PyUnresolvedReferences
    for x in dummies.columns:
        df["{0}.{1}".format(column, x)] = dummies[x]
    df.drop(column, axis=1, inplace=True)


def encode_categorical_index(df, column):
    """
        Transform a categorical value in indexes (for labels)
    """
    encoder = preprocessing.LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    return encoder.classes_


def encode_continuous_zscore(df, column):
    """
        Enconde continuous columns as zscores
    """
    df[column] = (df[column] - df[column].mean()) / df[column].std()


def to_xy(df, target):
    """
        处理成tensor张量
    """
    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if \
        hasattr(target_type, '__iter__') else target_type
    # Encode to int for classification, float otherwise. TF likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(df[target])
        return df.values.astype(
            np.float32), dummies.values.astype(np.float32)
    else:
        # Regression
        return df.values.astype(np.float32), df.as_matrix(
            [target]).astype(np.float32)


def equalize_columns(df1, df2):
    """
        Insert missing columns from df1 on df2
    """
    for column in df1.columns:
        if column not in df2.columns:
            df2[column] = pd.Series(0.5,
                                    index=df2.index)

    return df2


def outcome_to_type(df):
    for i, row in df.iterrows():
        df.set_value(i, 'outcome', ATTACK_DICT[row['outcome']])

    return df


def setup_data(training_data, test_data, sample=None):
    # Reading training set
    df_train = pd.read_csv(training_data, header=None)
    if sample:
        df_train = df_train.sample(frac=sample)
    logger.info("Training with {0} rows.".format(len(df_train)))

    # Reading validation set
    df_test = pd.read_csv(test_data, header=None)
    logger.info("Testing with {0} rows.".format(len(df_test)))

    # Dropping NaNs for now
    df_train.dropna(inplace=True, axis=1)
    df_test.dropna(inplace=True, axis=1)
    df_train.columns = COLUMNS
    df_test.columns = COLUMNS
    # df_test.columns = COLUMNS[0:-1]
    # df_train['outcome'] = df_train['outcome'].apply(
    #                         lambda o: o[:-1])
    # df_test['outcome'] = df_test['outcome'].apply(lambda o: o[:-1])

    df_train.drop('difficulty', inplace=True, axis=1)  # For NSL-KDD
    df_test.drop('difficulty', inplace=True, axis=1)  # For NSL-KDD
    df_train.drop('wrong_fragment', inplace=True, axis=1)  # For NSL-KDD
    df_test.drop('wrong_fragment', inplace=True, axis=1)  # For NSL-KDD
    df_train.drop('urgent', inplace=True, axis=1)  # For NSL-KDD
    df_test.drop('urgent', inplace=True, axis=1)  # For NSL-KDD
    df_train.drop('hot', inplace=True, axis=1)  # For NSL-KDD
    df_test.drop('hot', inplace=True, axis=1)  # For NSL-KDD
    df_train.drop('num_failed_logins', inplace=True, axis=1)  # For NSL-KDD
    df_test.drop('num_failed_logins', inplace=True, axis=1)  # For NSL-KDD
    df_train.drop('logged_in', inplace=True, axis=1)  # For NSL-KDD
    df_test.drop('logged_in', inplace=True, axis=1)  # For NSL-KDD
    df_train.drop('num_compromised', inplace=True, axis=1)  # For NSL-KDD
    df_test.drop('num_compromised', inplace=True, axis=1)  # For NSL-KDD
    df_train.drop('root_shell', inplace=True, axis=1)  # For NSL-KDD
    df_test.drop('root_shell', inplace=True, axis=1)  # For NSL-KDD
    df_train.drop('su_attempted', inplace=True, axis=1)  # For NSL-KDD
    df_test.drop('su_attempted', inplace=True, axis=1)  # For NSL-KDD
    df_train.drop('num_root', inplace=True, axis=1)  # For NSL-KDD
    df_test.drop('num_root', inplace=True, axis=1)  # For NSL-KDD
    df_train.drop('num_file_creations', inplace=True, axis=1)  # For NSL-KDD
    df_test.drop('num_file_creations', inplace=True, axis=1)  # For NSL-KDD
    df_train.drop('num_shells', inplace=True, axis=1)  # For NSL-KDD
    df_test.drop('num_shells', inplace=True, axis=1)  # For NSL-KDD
    df_train.drop('num_access_files', inplace=True, axis=1)  # For NSL-KDD
    df_test.drop('num_access_files', inplace=True, axis=1)  # For NSL-KDD
    df_train.drop('num_outbound_cmds', inplace=True, axis=1)  # For NSL-KDD
    df_test.drop('num_outbound_cmds', inplace=True, axis=1)  # For NSL-KDD
    df_train.drop('is_host_login', inplace=True, axis=1)  # For NSL-KDD
    df_test.drop('is_host_login', inplace=True, axis=1)  # For NSL-KDD
    df_train.drop('is_guest_login', inplace=True, axis=1)  # For NSL-KDD
    df_test.drop('is_guest_login', inplace=True, axis=1)  # For NSL-KDD


    # Classify all attacks together for now
    # df_train['outcome'] = df_train['outcome'].apply(
    #                     lambda o: 'normal' if o == 'normal' else 'attack')
    #
    # df_test['outcome'] = df_test['outcome'].apply(
    #                     lambda o: 'normal' if o == 'normal' else 'attack')
    df_train['outcome'] = df_train['outcome'].apply(
                            lambda row:ATTACK_DICT[row])
    df_test['outcome'] = df_test['outcome'].apply(
                            lambda row: ATTACK_DICT[row])

    df_train = df_train[df_train['outcome'].isin(['normal', 'probe','dos'])]
    df_test = df_test[df_test['outcome'].isin(['normal', 'probe', 'dos'])]

    # Treating the features (Training set)
    encode_continuous_zscore(df_train, 'duration')
    encode_one_hot(df_train, 'protocol_type')
    encode_one_hot(df_train, 'service')
    encode_one_hot(df_train, 'flag')
    encode_continuous_zscore(df_train, 'src_bytes')
    encode_continuous_zscore(df_train, 'dst_bytes')
    encode_one_hot(df_train, 'land')
    # encode_continuous_zscore(df_train, 'wrong_fragment')
    # encode_continuous_zscore(df_train, 'urgent')
    # encode_continuous_zscore(df_train, 'hot')
    # encode_continuous_zscore(df_train, 'num_failed_logins')
    # encode_one_hot(df_train, 'logged_in')
    # encode_continuous_zscore(df_train, 'num_compromised')
    # encode_continuous_zscore(df_train, 'root_shell')
    # encode_continuous_zscore(df_train, 'su_attempted')
    # encode_continuous_zscore(df_train, 'num_root')
    # encode_continuous_zscore(df_train, 'num_file_creations')
    # encode_continuous_zscore(df_train, 'num_shells')
    # encode_continuous_zscore(df_train, 'num_access_files')
    # encode_continuous_zscore(df_train, 'num_outbound_cmds')
    # encode_one_hot(df_train, 'is_host_login')
    # encode_one_hot(df_train, 'is_guest_login')
    encode_continuous_zscore(df_train, 'count')
    encode_continuous_zscore(df_train, 'srv_count')
    encode_continuous_zscore(df_train, 'serror_rate')
    encode_continuous_zscore(df_train, 'srv_serror_rate')
    encode_continuous_zscore(df_train, 'rerror_rate')
    encode_continuous_zscore(df_train, 'srv_rerror_rate')
    encode_continuous_zscore(df_train, 'same_srv_rate')
    encode_continuous_zscore(df_train, 'diff_srv_rate')
    encode_continuous_zscore(df_train, 'srv_diff_host_rate')
    encode_continuous_zscore(df_train, 'dst_host_count')
    encode_continuous_zscore(df_train, 'dst_host_srv_count')
    encode_continuous_zscore(df_train, 'dst_host_same_srv_rate')
    encode_continuous_zscore(df_train, 'dst_host_diff_srv_rate')
    encode_continuous_zscore(df_train, 'dst_host_same_src_port_rate')
    encode_continuous_zscore(df_train, 'dst_host_srv_diff_host_rate')
    encode_continuous_zscore(df_train, 'dst_host_serror_rate')
    encode_continuous_zscore(df_train, 'dst_host_srv_serror_rate')
    encode_continuous_zscore(df_train, 'dst_host_rerror_rate')
    encode_continuous_zscore(df_train, 'dst_host_srv_rerror_rate')
    encode_categorical_index(df_train, 'outcome')

    # Treating the features (Validation set)
    encode_continuous_zscore(df_test, 'duration')
    encode_one_hot(df_test, 'protocol_type')
    encode_one_hot(df_test, 'service')
    encode_one_hot(df_test, 'flag')
    encode_continuous_zscore(df_test, 'src_bytes')
    encode_continuous_zscore(df_test, 'dst_bytes')
    encode_one_hot(df_test, 'land')
    # encode_continuous_zscore(df_test, 'wrong_fragment')
    # encode_continuous_zscore(df_test, 'urgent')
    # encode_continuous_zscore(df_test, 'hot')
    # encode_continuous_zscore(df_test, 'num_failed_logins')
    # encode_one_hot(df_test, 'logged_in')
    # encode_continuous_zscore(df_test, 'num_compromised')
    # encode_continuous_zscore(df_test, 'root_shell')
    # encode_continuous_zscore(df_test, 'su_attempted')
    # encode_continuous_zscore(df_test, 'num_root')
    # encode_continuous_zscore(df_test, 'num_file_creations')
    # encode_continuous_zscore(df_test, 'num_shells')
    # encode_continuous_zscore(df_test, 'num_access_files')
    # encode_continuous_zscore(df_test, 'num_outbound_cmds')
    # encode_one_hot(df_test, 'is_host_login')
    # encode_one_hot(df_test, 'is_guest_login')
    encode_continuous_zscore(df_test, 'count')
    encode_continuous_zscore(df_test, 'srv_count')
    encode_continuous_zscore(df_test, 'serror_rate')
    encode_continuous_zscore(df_test, 'srv_serror_rate')
    encode_continuous_zscore(df_test, 'rerror_rate')
    encode_continuous_zscore(df_test, 'srv_rerror_rate')
    encode_continuous_zscore(df_test, 'same_srv_rate')
    encode_continuous_zscore(df_test, 'diff_srv_rate')
    encode_continuous_zscore(df_test, 'srv_diff_host_rate')
    encode_continuous_zscore(df_test, 'dst_host_count')
    encode_continuous_zscore(df_test, 'dst_host_srv_count')
    encode_continuous_zscore(df_test, 'dst_host_same_srv_rate')
    encode_continuous_zscore(df_test, 'dst_host_diff_srv_rate')
    encode_continuous_zscore(df_test, 'dst_host_same_src_port_rate')
    encode_continuous_zscore(df_test, 'dst_host_srv_diff_host_rate')
    encode_continuous_zscore(df_test, 'dst_host_serror_rate')
    encode_continuous_zscore(df_test, 'dst_host_srv_serror_rate')
    encode_continuous_zscore(df_test, 'dst_host_rerror_rate')
    encode_continuous_zscore(df_test, 'dst_host_srv_rerror_rate')
    encode_categorical_index(df_test, 'outcome')

    # Some NaNs appear again after the previous ops
    df_train.dropna(inplace=True, axis=1)
    df_test.dropna(inplace=True, axis=1)

    # Insert missing columns
    df_train = equalize_columns(df_test, df_train)
    df_test = equalize_columns(df_train, df_test)

    return df_train, df_test


def train_model(x_train, y_train, x_val, y_val):
    # ANN definition
    model = tf.contrib.keras.models.Sequential()
    model.add(tf.contrib.keras.layers.Dense(L1_HIDDEN,
                                            input_dim=x_train.shape[1],
                                            kernel_initializer='normal',
                                            activation='relu'))
    model.add(tf.contrib.keras.layers.Dropout(0.3))
    model.add(tf.contrib.keras.layers.Dense(L2_HIDDEN,
                                            activation='relu'))
    model.add(tf.contrib.keras.layers.Dropout(0.2))
    model.add(tf.contrib.keras.layers.Dense(L3_HIDDEN,
                                            activation='relu'))
    model.add(tf.contrib.keras.layers.Dense(L4_HIDDEN,
                                            activation='relu'))
    model.add(tf.contrib.keras.layers.Dropout(0.1))
    model.add(tf.contrib.keras.layers.Dense(L5_HIDDEN,
                                            activation='relu'))
    model.add(tf.contrib.keras.layers.Dense(y_train.shape[1],
                                            activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    # Early stopping for low loss, make testing easier
    monitor = tf.contrib.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       min_delta=1e-5,
                                                       patience=5,
                                                       verbose=1,
                                                       mode='auto')
    # model.fit(x_train, y_train, validation_data=(x_val, y_val),
    #           callbacks=[], verbose=1, epochs=20)
    # model.save_weights("s_model_321_20.h5")

    model.load_weights('s_model_321_20.h5')


    return model


def run_model(model, x_test, y_test):
    pred = model.predict(x_test)
    pred = np.argmax(pred, axis=1)
    corr = np.argmax(y_test, axis=1)

    logger.info("Accuracy: {}".format(metrics.accuracy_score(corr, pred)))
    logger.info("Recall: {}".format(metrics.recall_score(
                                                    corr, pred, average=None)))
    logger.info("Precision: {}".format(metrics.precision_score(corr, pred,
                                                               average=None)))
    logger.info("F1 Score: {}".format(metrics.f1_score(corr, pred,
                                                       average=None)))

    return corr, pred


def main():
    df_train, df_test = setup_data(TRAINING_FILE, TEST_FILE)

    # Convert to TF format
    x, y = to_xy(df_train, 'outcome')
    x_test, y_test = to_xy(df_test, 'outcome')

    # Separate some of dataset for validation
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

    model = train_model(x_train, y_train, x_val, y_val)
    corr, pred = run_model(model, x_test, y_test)

    # Plot Confusion Matrix as heatmap
    # TODO: Insert labels on confusion matrix
    confusion = metrics.confusion_matrix(corr, pred)
    df_cm = pd.DataFrame(confusion, index=[i for i in range(len(confusion))],
                         columns=[i for i in range(len(confusion))])
    plt.figure(figsize=(20, 15))
    plot = sn.heatmap(df_cm, annot=True)
    fig = plot.get_figure()
    fig.savefig("confusion_2.png")


if __name__ == '__main__':
    main()
