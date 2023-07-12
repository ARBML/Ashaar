import yaml

fname = "config/gpt-cls-tash-proc.yml"

stream = open(fname, 'r')
data = yaml.load(stream,  Loader=yaml.FullLoader)

for i in range(0, 10):
    data['n_layer'] = i
    data['log_directory'] = f'log_dir_cls_{i}_tash_proc'
    data['max_steps'] = 5000
    with open(f"config/gpt-cls-{i}-tash-proc.yml", 'w') as yaml_file:
        yaml_file.write( yaml.dump(data, default_flow_style=False))
