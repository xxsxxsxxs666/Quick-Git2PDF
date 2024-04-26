import yaml
import argparse
import pathlib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=None)

    args = parser.parse_args()
    assert args.config_path is not None, "config_path is None"
    name = pathlib.Path(args.config_path).name
    parent = pathlib.Path(args.config_path).parent
    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # full_key = ['I_M', 'I_A', 'CS_M', 'CS_A', 'CS_DL']
    # config['infer']['key'] = ['I_M', 'I_A', 'CS_M', 'CS_A', 'CS_DL']
    # config['train']['key'] = ['I_M', 'I_A', 'CS_M', 'CS_A', 'CS_DL']
    # config['model']['in_channels'] = len(config['infer']['key'])
    # config['model']['model_parameter']['num_input_channels'] = len(config['infer']['key'])
    # with open(parent / name.replace(".yaml", f"_new_0.yaml"), 'w') as f:
    #     yaml.dump(config, f, encoding='utf-8', allow_unicode=True)
    #
    # config['infer']['key'] = ['I_M', 'CS_M', 'CS_A', 'CS_DL']
    # config['train']['key'] = ['I_M', 'CS_M', 'CS_A', 'CS_DL']
    # config['model']['in_channels'] = len(config['infer']['key'])
    # config['model']['model_parameter']['num_input_channels'] = len(config['infer']['key'])
    # with open(parent / name.replace(".yaml", f"_new_1.yaml"), 'w') as f:
    #     yaml.dump(config, f, encoding='utf-8', allow_unicode=True)
    #
    # config['infer']['key'] = ['I_M', 'I_A', 'CS_A', 'CS_DL']
    # config['train']['key'] = ['I_M', 'I_A', 'CS_A', 'CS_DL']
    # config['model']['in_channels'] = len(config['infer']['key'])
    # config['model']['model_parameter']['num_input_channels'] = len(config['infer']['key'])
    # with open(parent / name.replace(".yaml", f"_new_2.yaml"), 'w') as f:
    #     yaml.dump(config, f, encoding='utf-8', allow_unicode=True)
    #
    # config['infer']['key'] = ['I_M', 'I_A', 'CS_M', 'CS_DL']
    # config['train']['key'] = ['I_M', 'I_A', 'CS_M', 'CS_DL']
    # config['model']['in_channels'] = len(config['infer']['key'])
    # config['model']['model_parameter']['num_input_channels'] = len(config['infer']['key'])
    # with open(parent / name.replace(".yaml", f"_new_3.yaml"), 'w') as f:
    #     yaml.dump(config, f, encoding='utf-8', allow_unicode=True)

    # config['infer']['key'] = ['I_M', 'I_A', 'CS_M', 'CS_A']
    # config['train']['key'] = ['I_M', 'I_A', 'CS_M', 'CS_A']
    # config['model']['in_channels'] = len(config['infer']['key'])
    # config['model']['model_parameter']['num_input_channels'] = len(config['infer']['key'])
    # with open(parent / name.replace(".yaml", f"_new_4.yaml"), 'w') as f:
    #     yaml.dump(config, f, encoding='utf-8', allow_unicode=True)

    # config['infer']['key'] = ['I_M']
    # config['train']['key'] = ['I_M']
    # config['model']['in_channels'] = len(config['infer']['key'])
    # config['model']['model_parameter']['num_input_channels'] = len(config['infer']['key'])
    # with open(parent / name.replace(".yaml", f"_new_I_M.yaml"), 'w') as f:
    #     yaml.dump(config, f, encoding='utf-8', allow_unicode=True)

    config['infer']['key'] = ['I_M', 'CS_M', 'CS_DL']
    config['train']['key'] = ['I_M', 'CS_M', 'CS_DL']
    config['model']['in_channels'] = len(config['infer']['key'])
    config['model']['model_parameter']['num_input_channels'] = len(config['infer']['key'])
    with open(parent / name.replace(".yaml", f"_new_no_a.yaml"), 'w') as f:
        yaml.dump(config, f, encoding='utf-8', allow_unicode=True)

    config['infer']['key'] = ['I_M', 'I_A', 'CS_A', 'CS_DL']
    config['train']['key'] = ['I_M', 'CS_M', 'CS_A', 'CS_DL']
    config['model']['in_channels'] = len(config['infer']['key'])
    config['model']['model_parameter']['num_input_channels'] = len(config['infer']['key'])
    with open(parent / name.replace(".yaml", f"_new_MMAD.yaml"), 'w') as f:
        yaml.dump(config, f, encoding='utf-8', allow_unicode=True)







