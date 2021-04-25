import os
from flask import Flask, request
import pandas as pd
import json
import pickle
import numpy as np

from tools import test_quality


app = Flask(__name__)


@app.route('/')
def hello():
    return "This is an app with RWSM model. Available methods: train"


def train(base_args):
    args_str = " ".join(["--" + arg_name + "=" + str(arg_val) for arg_name, arg_val in base_args.items()])
    print("Model fitted with parameters: " + args_str)
    os.system('python3.7 train.py {}'.format(args_str))
    with open(base_args['config_path'], 'rb') as f:
        config = json.load(f)
    df_res = calc_stats(prediction_pkl_path=base_args['save_path'] + base_args['model_type'] + "_val_pred.pkl",
                        data_path=base_args['val_data_path'],
                        losses_path=base_args['save_path'] + base_args["model_type"] + '_losses.csv',
                        config=config)
    return df_res


@app.route('/demo', methods=['GET'])
def train_demo_model():
    base_args = {
        'model_type': "base",
        'train_data_path': "data/input_metabric/metabric_preprocessed_cv_0_train.pkl",
        'val_data_path': "data/input_metabric/metabric_preprocessed_cv_0_test.pkl",
        'custom_bottom_function_name': "metabric_main_network",
        'verbose': 1,
        'save_path': "data/test_flask/",
        'save_prediction': True,
        'save_losses': True,
        'config_path': "configs/default_base.json"
    }
    df_res = train(base_args)
    return "Quality stats at last epoch: " + str(df_res.tail(1).T.to_dict())


@app.route('/train', methods=['POST', 'GET'])
def train_model():
    base_args = {
        'model_type': "base",
        'train_data_path': "data/input_metabric/metabric_preprocessed_cv_0_train.pkl",
        'val_data_path': "data/input_metabric/metabric_preprocessed_cv_0_test.pkl",
        'custom_bottom_function_name': "metabric_main_network",
        'verbose': 1,
        'save_path': "data/test_flask/",
        'save_prediction': True,
        'save_losses': True
    }
    data = request.get_json(force=True)
    for par_name, par_val in data.items():
        if par_name in base_args:
            base_args.update({par_name: par_val})
    # if config is not given in request, use default config for this type of model
    if 'config_path' not in data:
        base_args.update({'config_path': "configs/default_{}.json".format(base_args['model_type'])})
    else:
        base_args.update({'config_path': data['config_path']})
    df_res = train(base_args)
    return "Quality stats at last epoch: " + str(df_res.tail(1).T.to_dict())


def calc_stats(prediction_pkl_path, data_path, losses_path, config):
    with open(prediction_pkl_path, 'rb') as f:
        predictions = pickle.load(f)
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    df_losses = pd.read_csv(losses_path, index_col=0)
    q = []
    for pred in predictions:
        q.append(
            test_quality(
                t_true=data['t'], y_true=data['y'], pred=pred, time_grid=np.array(config['time_grid']),
                concordance_at_t=None, plot=False
            )
        )
    df_all_q = pd.concat(q)
    df_all_q.reset_index(drop=True, inplace=True)
    df_all_q = pd.concat([df_losses, df_all_q], axis=1)
    return df_all_q


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
