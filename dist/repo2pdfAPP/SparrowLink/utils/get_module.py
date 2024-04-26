import monai.networks.nets as net
from monai.networks.layers import Norm
from model.DSnet import denseUnet3d
from model.denseunet_skip import SkipDenseUNet
from model.merge import Merge
from model.CS2net import CSNet3D
from model.liujm import SegNet
from model.liujmMS import SegNetMultiScale
import model
from monai.networks.layers import Act
import torch.optim
import monai.losses
import monai.metrics
import loss_zoo
import metric_zoo


def get_default_net(dic):
    net_name = dic['model']['name']
    if net_name == 'UNet':
        return net.unet.UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,)
    elif net_name == 'Vnet':
        return net.vnet.VNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            dropout_prob=0.5,
            dropout_dim=3,
        )
    elif net_name == 'DenseUnet':
        return denseUnet3d(
            num_input=1,
            out_channels=2,
        )
    elif net_name == 'SkipDenseUNet':
        return SkipDenseUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,)
    elif net_name == 'Merge':
        return Merge(spatial_dims=3,
                     in_channels=2,
                     out_channels=2,
                     channels=(8, 8, 8, 8),
                     strides=(2, 2, 2),
                     num_res_units=2,
                     norm=Norm.BATCH,
                     repeat=2)
    elif net_name == 'CS2net':
        return CSNet3D(in_channels=1, out_channels=2)
    elif net_name == 'liujm':
        return SegNet(in_channels=1,
                      out_channels=2)
    elif net_name == 'liujmMS':
        return SegNetMultiScale(in_channels=1,
                                out_channels=2)
    else:
        RuntimeError('The net isn"t available in default net mode yet')


def get_concrete_net(dic):
    """
    The Net parameter are defined in config, Now only nets in monai are included
    :param dic: cfg.dic, loaded from config.yaml, containing the concrete parameter of model frame
    :return: instantiate model
    """
    net_name = dic['model']['name']
    if net_name == 'nnunetv2':
        return get_nnunet_architecture(**dic['model']['model_parameter'])
    assert net_name in dir(net) + dir(model), "The net isn't available yet"
    if net_name in dir(model):
        config_model = eval('model.'+net_name)
    else:
        config_model = eval('net.'+net_name)
    return config_model(**dic['model']['model_parameter'])


def get_nnunet_architecture(model_training_output_dir, checkpoint_path, num_input_channels, enable_deep_supervision=False, load_checkpoints=False):
    """
    The Net parameter are defined in config, Now only nets in monai are included
    :param nnunet_dic: having
    """
    import json
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from torch._dynamo import OptimizedModule
    with open(model_training_output_dir + '/dataset.json', 'r') as f:
        dataset_json = json.load(f)
    with open(model_training_output_dir + 'plans.json', 'r') as f:
        plans = json.load(f)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(checkpoint['init_args']['configuration'])
    network = get_network_from_plans(plans_manager, dataset_json, configuration_manager,
                                     num_input_channels, deep_supervision=enable_deep_supervision)
    # from IPython import embed; embed(colors='linux')
    if load_checkpoints:
        if not isinstance(network, OptimizedModule):
            network.load_state_dict(checkpoint['network_weights'])
        else:
            network._orig_mod.load_state_dict(checkpoint['network_weights'])
    return network


def get_optimizer(name):
    """
    return uninstantiate optimizer, now only support optimizer in torch.optim
    :param name:
    :return:
    """
    assert name in dir(torch.optim), 'this optimizer is not supported yet'
    return eval('torch.optim.'+name)


def get_lr_scheduler(name):
    """
    return uninstantiate optimizer, now only support optimizer in torch.optim
    :param name:
    :return:
    """
    assert name in dir(torch.optim.lr_scheduler), 'this learning scheduler is not supported yet'
    return eval('torch.optim.lr_scheduler.'+name)


def get_loss(dic):
    """
    :param dic: {name:DiceMetric, parameter:{}}
    :return:
    """
    if dic['name'] == None:
        return None
    # get the model if name is in loss_zoo or monai.losses, donot use eval and dir()
    assert dic['name'] in dir(monai.losses) + dir(loss_zoo), 'this loss is not supported yet'
    if dic['name'] in dir(loss_zoo):
        loss = eval('loss_zoo.'+dic['name'])
    else:
        loss = eval('monai.losses.'+dic['name'])
    return loss(**dic['parameter'])


def get_metric(dic):
    """
    :param dic: {name:DiceMetric, parameter:{}}
    :return:
    """
    assert dic['name'] in dir(monai.metrics) + dir(metric_zoo), 'this metric is not supported yet'
    if dic['name'] in dir(metric_zoo):
        metrics = eval('metric_zoo.'+dic['name'])
    else:
        metrics = eval('monai.metrics.'+dic['name'])
    return metrics(**dic['parameter'])


if __name__ == "__main__":
    # from Config import Config
    # print('-------------------------')
    # cfg = Config('../configs/config.yaml')
    # config_model = get_concrete_net(cfg.dic)
    # default_model = get_default_net(cfg.dic)
    # optimizer = get_optimizer(cfg.dic['train']['optimizer']['name'])
    # lr_scheduler = get_lr_scheduler(cfg.dic['train']['lr_scheduler']['name'])
    # loss = get_loss(cfg.dic['train']['train_loss'])
    # metric = get_metric(cfg.dic['train']['val_metric'])
    model = net.densenet.DenseNet(spatial_dims=3, in_channels=1, out_channels=2)
    x = torch.randn(1, 1, 64, 64, 64)
    y = model(x)
    breakpoint()
    pass

"""m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)"""
