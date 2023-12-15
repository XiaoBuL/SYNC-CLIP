import random
from collections import defaultdict

def split_dataset_by_label(data_source):
    output = defaultdict(list)

    for item in data_source:
        output[item.label].append(item)

    return output

def Generate_Fewshot_Dataset(*data_sources, num_shots=-1, repeat=True):
    if num_shots < 1:
        if len(data_sources) == 1:
            return data_sources[0]
        return data_sources

    print(f'Creating a {num_shots}-shot dataset')

    output = []

    for data_source in data_sources:
        tracker = split_dataset_by_label(data_source)
        dataset = []

        for label, items in tracker.items():
            if len(items) >= num_shots:
                sampled_items = random.sample(items, num_shots)
            else:
                if repeat:
                    sampled_items = random.choices(items, k=num_shots)
                else:
                    sampled_items = items
            dataset.extend(sampled_items)

        output.append(dataset)

    if len(output) == 1:
        return output[0]

    return output