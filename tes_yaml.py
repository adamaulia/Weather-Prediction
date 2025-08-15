import yaml
config_file = 'config.yaml'
with open(config_file, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
    
# print(config['export_weather'])

# print(config.get('historical').get('include_hourly'))
# print(config.get('historical.include_hourly', False))

# if config.get('export_weather').get('enabled'):
#     print('ok')
# else :
#     print('not ok')

if config['export_weather']['enabled']:
    print('ok')
else :
    print('not ok')
