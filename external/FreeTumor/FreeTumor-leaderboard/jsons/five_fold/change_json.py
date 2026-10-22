import json
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import *


def run():
    file = open('./kits_splits.json')
    file = json.load(file)
    print(file[4].keys())
    ls = file[4]['train']

    original = open('dataset_kits_all.json')
    original = json.load(original)

    y = original.copy()
    y['training'] = []
    y['validation'] = []
    y['numTraining'] = 0

    for i in tqdm(original['training']):
        name = i['label'].split('/')[-1][:-7]

        if name in ls:
            y['training'].append(i)
            y['numTraining'] += 1
            print('add:', y['numTraining'], i)
        else:
            y['validation'].append(i)

        save_json(y, './dataset_kits_split4.json', sort_keys=True)


if __name__ == '__main__':
    run()