

def setup_experiment(exp_name, exp_params, net_params, ckpts_folder='saved_checkpoints'):

    exp_desc = {
        'header': {},
        'expparams': exp_params,
        'netparams': net_params
    }

    import time
    exp_desc['header']['timestamp'] = time.time()

    import os
    model_folder = os.path.join(ckpts_folder, exp_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    descriptor_file = os.path.join(model_folder, 'description.yaml')
    if os.path.exists(descriptor_file):
        return False

    import yaml
    with open(descriptor_file, 'w') as file:
        _ = yaml.dump(exp_desc, file)
    return True
