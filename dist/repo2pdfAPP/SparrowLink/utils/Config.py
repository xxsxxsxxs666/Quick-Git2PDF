import codecs
import os
from typing import Any, Dict, Generic
from utils.get_module import (
    get_optimizer,
    get_lr_scheduler,
    get_default_net,
    get_concrete_net,
    get_loss,
    get_metric,
)
import yaml
import torch
import json
from pre_processing.checking import DatasetinformationExtractor
from data.loader import prepare_datalist
import pathlib
from typing import List, Sequence, Tuple, Union


class Config(object):
    """
    1. load yaml and transfer it into dic
    2. create model, loss, optimizer, lr_scheduler, etc. by  function:creat_training_require(self)
    3. config.yaml must contain
    """
    def __init__(self,
                 path: str,
                 learning_rate: float = None,
                 batch_size: int = None,
                 iters: int = None,
                 seed: int = None,
                 experiments_path: str = None,
                 mode: str = None,
                 device: str = None,
                 img_path: str = None,
                 I_M: str = None,
                 I_A: str = None,
                 CS_M: str = None,
                 CS_A: str = None,
                 CS_DLGT: str = None,
                 CS_DL: str = None,
                 CS_W: str = None,
                 second_stage_key: Sequence[str] = None,
                 select_file: str = None,
                 label_path: str = None,
                 output_path: str = None,
                 val_set: str = None,
                 train_set: str = None,
                 persist_path: str = None,
                 pretrain_weight_path: str = None,
                 dataset_information: str = None,
                 ):
        """
        :param path: config.yaml path
        :param learning_rate: from parser, the priority of parser is higher than config.yaml
        :param batch_size: from parser
        :param iters: means epoch during training, quite confusing hhh, there is a iter_in_epoch in trainer, default:100
        :param seed: from parser
        :param experiments_path: almost all result and midiate output is in here
        :param mode: train, test, infer are available here
        :param device: cpu or gpu. ddp is not supported yet
        :param img_path: for the first stage training
        :param I_M:
        :param I_A:
        :param CS_M:
        :param CS_A:
        :param CS_DLGT:
        :param CS_DL:
        :param CS_W:

        """
        if not path:
            raise ValueError('Please specify the configuration file path.')

        if not os.path.exists(path):
            raise FileNotFoundError('File {} does not exist'.format(path))

        if path.endswith('yml') or path.endswith('yaml'):
            self.dic = self._parse_from_yaml(path)
        else:
            raise RuntimeError('Config file should in yaml format!')

        # Because Parse will change three hyper-parameters if used, so use update function to update self.dic
        # also if Parse has opts, use this function to update self.dci
        self.update(
            learning_rate=learning_rate,
            batch_size=batch_size,
            iters=iters,
            seed=seed,
            experiments_path=experiments_path,
            mode=mode,
            device=device,
            img_path=img_path,
            I_M=I_M,
            I_A=I_A,
            CS_M=CS_M,
            CS_A=CS_A,
            CS_DLGT=CS_DLGT,
            CS_DL=CS_DL,
            CS_W=CS_W,
            select_file=select_file,
            label_path=label_path,
            output_path=output_path,
            val_set=val_set,
            train_set=train_set,
            persist_path=persist_path,
            pretrain_weight_path=pretrain_weight_path,
        )
        self.mode = self.dic['mode']
        self.seed = self.dic['seed']
        if self.dic.get('device'):
            self.device = torch.device(self.dic['device'])
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.use_tensorboard = None
        self.show_val_index = None
        self.learning_rate = None
        self.seed = None
        self.model = None
        self.optimizer_init = None
        self.lr_scheduler_init = None
        self.train_loss = None
        self.val_metric = None
        self.start_epoch = None
        self.lr_scheduler_init = None
        self.infer_metric = None
        self.train_sw_overlap = self.dic['train'].get('overlap') if self.dic['train'].get('overlap') else 0.25
        self.train_sw_batch_size = self.dic['train'].get('sw_batch_size') if self.dic['train'].get('sw_batch_size') else 4
        self.train_mirror_axes = self.dic['train'].get('mirror_axes') if self.dic['train'].get('mirror_axes') else []
        self.infer_sw_overlap = self.dic['infer'].get('overlap') if self.dic['infer'].get('overlap') else 0.25
        self.infer_sw_batch_size = self.dic['infer'].get('sw_batch_size') if self.dic['infer'].get('sw_batch_size') else 4
        self.infer_mirror_axes = self.dic['infer'].get('mirror_axes') if self.dic['infer'].get('mirror_axes') else []

        # data related
        self.train_img_path = None
        self.train_label_path = None
        self.val_set = None
        self.train_set = None
        self.test_img_path = None
        self.label_path = None
        self.infer_img_path = None
        self.persist_path = None

        # output related
        self.infer_output_path = None
        self.test_output_path = None
        self.experiments_path = self.dic['experiments_path']
        pathlib.Path(experiments_path).mkdir(parents=True, exist_ok=True)
        # second_stage related
        self.I_M = self.dic[self.mode]['loader'].get('I_M')
        self.I_A = self.dic[self.mode]['loader'].get('I_A')
        self.CS_M = self.dic[self.mode]['loader'].get('CS_M')
        self.CS_A = self.dic[self.mode]['loader'].get('CS_A')
        self.CS_DLGT = self.dic[self.mode]['loader'].get('CS_DLGT')
        self.CS_DL = self.dic[self.mode]['loader'].get('CS_DL')
        self.CS_W = self.dic[self.mode]['loader'].get('CS_W')
        self.select_file = self.dic[self.mode]['loader'].get('select_file')
        self.select_file = self.dic[self.mode]['loader'].get('select_file')
        self.second_stage_key = self.dic[self.mode].get('key')

        # transformation related
        # load dic from dataset_information.json, 和模型文件夹同级
        dataset_information_path = pathlib.Path(self.experiments_path).parent / "dataset_properties.json" \
            if dataset_information is None else pathlib.Path(dataset_information)
        # print(dataset_information_path)
        if not dataset_information_path.exists():
            if img_path is None:
                img_path = self.I_M  # for second_stage
            assert img_path is not None and label_path is not None, "img_path and label_path must be " \
                                                                    "specified when no dataset_information.json exists"
            data_list = prepare_datalist(image_file=img_path, label_file=label_path, split_mode="all")
            information_extractor = DatasetinformationExtractor(
                data_list=data_list,
                image_key="image",
                label_key="label",
                preprocessed_output_folder=dataset_information_path.parent,
                multi_process=True,
                json_name="dataset_properties.json",
            )
            self.dataset_information = information_extractor.run(overwrite_existing=True)
            with open(str(dataset_information_path), 'w') as f:
                json.dump(self.dataset_information, f)
        else:
            with open(str(dataset_information_path), 'r') as f:
                self.dataset_information = json.load(f)

        # update transformation in self.dic
        self.update_transformation()

    def update_transformation(self):
        self.dic["transform"]["spacing"] = self.dataset_information["median_spacings"]
        self.dic["transform"]["normalize"] = {"mean": self.dataset_information["foreground_intensity_properties"]["mean"],
                                              "std": self.dataset_information["foreground_intensity_properties"]["std"],
                                              "min": self.dataset_information["foreground_intensity_properties"]["percentile_00_5"],
                                              "max": self.dataset_information["foreground_intensity_properties"]["percentile_99_5"]
                                              }

    def creat_infer_require(self):
        self.infer_output_path = self.dic['infer']['output_path']
        self.use_tensorboard = self.dic['visual']['use_tensorboard']
        self.show_val_index = self.dic['visual']['show_image_index']
        self.infer_img_path = self.dic['infer']['loader'].get('img')

        if not os.path.exists(self.infer_output_path):
            os.makedirs(self.infer_output_path)
        if self.dic['model']['use_default']:
            self.model = get_default_net(self.dic)
            print(f"default_model is built!")
        else:
            self.model = get_concrete_net(self.dic)
            print(f"config_model is built!")
        if self.dic['infer'].get('pretrain_weight'):
            self.model.load_state_dict(torch.load(self.dic['infer']['pretrain_weight']))
        else:
            flag = self.dic["model"]["model_parameter"].get("load_checkpoints")
            print(flag)
            if not flag:
                raise RuntimeError('You have not writen pretrain_weight path in infer mode')

        self.print_infer_config()

    def creat_test_require(self):
        self.test_output_path = self.dic['test']['output_path']
        self.test_label_path = self.dic['test']['loader']['label']
        self.test_img_path = self.dic['test']['loader'].get('img')

        if not os.path.exists(self.test_output_path):
            os.makedirs(self.test_output_path)
        if self.dic['model']['use_default']:
            self.model = get_default_net(self.dic)
            print(f"default_model is built!")
        else:
            self.model = get_concrete_net(self.dic)
            print(f"config_model is built!")
        if self.dic['test'].get('pretrain_weight'):
            self.model.load_state_dict(torch.load(self.dic['test']['pretrain_weight']))
        else:
            raise RuntimeError('You have not writen pretrain_weight path in infer mode')
        if self.dic['test'].get('test_metric'):
            self.infer_metric = get_metric(self.dic['test']['test_metric'])
        else:
            raise RuntimeError('You have not write metric config')

        self.print_test_config()

    def creat_training_require(self):

        # -------------- Create Model, Loss, Metric, Optimizer -------------- #
        # ------------------------ using config ----------------------------- #
        # separate it from __init__ for better debug, and separate it from infer mode
        self.train_img_path = self.dic['train']['loader'].get('img')
        self.train_label_path = self.dic['train']['loader']['label']
        self.val_set = self.dic['train']['loader']['val_set'] if self.dic['train']['loader'].get('val_set') else None
        self.train_set = self.dic['train']['loader']['train_set'] if self.dic['train']['loader'].get('train_set') else None
        self.persist_path = self.dic['train']['loader']['persist'] \
            if self.dic['train']['loader'].get('persist') else None
        self.learning_rate = self.dic['learning_rate']
        print(self.dic['experiments_path'])
        if self.dic['model']['use_default']:
            self.model = get_default_net(self.dic)
            print(f"default_model is built!")
        else:
            self.model = get_concrete_net(self.dic)
            print(f"config_model is built!")
        if self.dic['train']['pretrain_weight']:
            self.model.load_state_dict(torch.load(self.dic['train']['pretrain_weight']))

        # you can delete optimizer in your config, and this config will not create optimizer
        if self.dic['train'].get('optimizer'):
            optimizer = get_optimizer(self.dic['train']['optimizer']['name'])  # here use config to definite module
            lr_scheduler = get_lr_scheduler(self.dic['train']['lr_scheduler']['name'])  # using torch directly is OK
            self.optimizer_init = optimizer(self.model.parameters(),
                                            lr=self.learning_rate, **self.dic['train']['optimizer']['parameter'])
            self.lr_scheduler_init = lr_scheduler(self.optimizer_init, **self.dic['train']['lr_scheduler']['parameter'])
            self.start_epoch = 0
            if self.dic['train']['continue_train']:  # use the epoch and lr in last experiment
                train_dic = torch.load(self.dic['train']['continue_train'])
                self.optimizer_init = optimizer([{'params': self.model.parameters(), 'initial_lr': train_dic['lr']}],
                                                lr=self.learning_rate, **self.dic['train']['optimizer']['parameter'])
                self.optimizer_init.load_state_dict(train_dic['optimizer_state'])
                self.start_epoch = train_dic['epoch']
                self.optimizer_to(self.optimizer_init, self.device)
                self.lr_scheduler_init = lr_scheduler(self.optimizer_init, last_epoch=self.start_epoch,
                                                      **self.dic['train']['lr_scheduler']['parameter'])

        else:
            raise RuntimeError('You have not write optimize config')
        if self.dic['train'].get('train_loss'):
            self.train_loss = get_loss(self.dic['train']['train_loss'])
        else:
            raise RuntimeError('You have not write loss config')
        if self.dic['train'].get('val_metric'):
            self.val_metric = get_metric(self.dic['train']['val_metric'])
        else:
            raise RuntimeError('You have not write metric config')
        self.print_train_config()

    def print_train_config(self):
        print(f"device: {self.device}")
        print(f"split_mode: {self.dic['train']['loader'].get('split_mode')}")
        print(f"pretrain_weight: {self.dic['train']['pretrain_weight']}")
        print(f"continue train: {self.dic['train']['continue_train']}")
        print('--------------model---------------------------')
        for k, v in self.dic['model'].items():
            print(f"{k}:{v}")
        print('--------------optimizer-----------------------')
        print(self.optimizer_init)
        print('--------------lr_scheduler--------------------')
        for k, v in self.dic['train']['lr_scheduler'].items():
            print(f"{k}:{v}")
        print('--------------train_loss--------------------')
        for k, v in self.dic['train']['train_loss'].items():
            print(f"{k}:{v}")
        print('--------------val_metrics--------------------')
        for k, v in self.dic['train']['val_metric'].items():
            print(f"{k}:{v}")
        print('--------------training_config information end-----------------')

    def save_config(self, save_path):
        with open(save_path, 'w') as file:
            yaml.dump(self.dic, file, indent=1, encoding='utf-8',allow_unicode=True)

    def print_test_config(self):
        print(f"device: {self.device}")
        print('----------------model---------------------------')
        for k, v in self.dic['model'].items():
            print(f"{k}:{v}")

        print('--------------val_metrics--------------------')
        for k, v in self.dic['test']['test_metric'].items():
            print(f"{k}:{v}")
        print('------------------path------------------------')
        print(f"test output is in {self.test_output_path}")
        print('--------------test_config information end-----------------')

    def print_infer_config(self):
        print(f"device: {self.device}")
        print('----------------model---------------------------')
        for k, v in self.dic['model'].items():
            print(f"{k}:{v}")
        print('--------------sliding windows--------------------')
        print(f"overlap: {self.infer_sw_overlap}")
        print(f"sw_batch_size: {self.infer_sw_batch_size}")
        print(f"mirroring: {self.infer_mirror_axes}")
        print('------------------path------------------------')
        print(f"test output is in {self.infer_output_path}")
        print('--------------infer_config information end-----------------')

    def _update_dic(self, dic, base_dic):
        """
        Update configs from dic based base_dic
        """
        base_dic = base_dic.copy()
        dic = dic.copy()

        if dic.get('_inherited_', True) == False:
            dic.pop('_inherited_')
            return dic

        for key, val in dic.items():
            if isinstance(val, dict) and key in base_dic:
                base_dic[key] = self._update_dic(val, base_dic[key])
            else:
                base_dic[key] = val
        dic = base_dic
        return dic

    def _parse_from_yaml(self, path: str):
        '''Parse a yaml file and build configs'''
        with codecs.open(path, 'r', 'utf-8') as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)

        if '_base_' in dic:
            cfg_dir = os.path.dirname(path)
            base_path = dic.pop('_base_')
            base_path = os.path.join(cfg_dir, base_path)
            base_dic = self._parse_from_yaml(base_path)
            dic = self._update_dic(dic, base_dic)
        return dic

    def update(self,
               learning_rate: float = None,
               batch_size: int = None,
               iters: int = None,
               seed: int = None,
               experiments_path: str = None,
               mode: str = None,
               device: str = None,
               img_path: str = None,
               I_M: str = None,
               I_A: str = None,
               CS_M: str = None,
               CS_A: str = None,
               CS_DLGT: str = None,
               CS_DL: str = None,
               CS_W: str = None,
               select_file: str = None,
               label_path: str = None,
               output_path: str = None,
               val_set: str = None,
               train_set: str = None,
               persist_path: str = None,
               pretrain_weight_path: str = None,
               ):
        '''Update configs'''
        if learning_rate:
            self.dic['learning_rate'] = learning_rate

        if batch_size:
            self.dic['batch_size'] = batch_size

        if iters:
            self.dic['iters'] = iters

        if seed and seed >= 0:
            self.dic['seed'] = seed

        if experiments_path:
            self.dic['experiments_path'] = experiments_path

        if mode:
            self.dic['mode'] = mode

        if device:
            self.dic['device'] = device

        if output_path:
            self.dic[self.dic['mode']]['output_path'] = output_path

        if img_path:
            self.dic[self.dic['mode']]['loader']['img'] = img_path

        if I_M:
            self.dic[self.dic['mode']]['loader']['I_M'] = I_M

        if I_A:
            self.dic[self.dic['mode']]['loader']['I_A'] = I_A

        if CS_M:
            self.dic[self.dic['mode']]['loader']['CS_M'] = CS_M

        if CS_A:
            self.dic[self.dic['mode']]['loader']['CS_A'] = CS_A

        if CS_DL:
            self.dic[self.dic['mode']]['loader']['CS_DL'] = CS_DL

        if CS_DLGT:
            self.dic[self.dic['mode']]['loader']['CS_DLGT'] = CS_DLGT

        if label_path:
            self.dic[self.dic['mode']]['loader']['label'] = label_path

        if val_set:
            self.dic['train']['loader']['val_set'] = val_set

        if train_set:
            self.dic['train']['loader']['train_set'] = train_set

        if select_file:
            self.dic[self.dic['mode']]['loader']['select_file'] = select_file

        if persist_path:
            self.dic[self.dic['mode']]['loader']['persist'] = persist_path

        if pretrain_weight_path:
            self.dic[self.dic['mode']]['pretrain_weight'] = pretrain_weight_path

        if CS_W:
            self.dic[self.dic['mode']]['loader']['CS_W'] = CS_W




    @property
    def batch_size(self) -> int:
        return self.dic.get('batch_size', 1)

    @property
    def iters(self) -> int:
        iters = self.dic.get('iters')
        if not iters:
            raise RuntimeError('No iters specified in the configuration file.')
        return iters

    @property
    def test_config(self) -> Dict:
        return self.dic.get('test_config', {})

    @property
    def export_config(self) -> Dict:
        return self.dic.get('export', {})

    def __str__(self) -> str:
        return yaml.dump(self.dic)

    def optimizer_to(self, optim, device):
        for param in optim.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)




