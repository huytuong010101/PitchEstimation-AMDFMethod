import os


def get_label(dir_path: str):
    """
    Load label form .lab file
    :param dir_path: Path to directory which include .lab file
    :return:
        lables: dictionary contain all label (silent, unvoice, voice time, f0 mean, f0 std)
    """
    label_names = [item for item in os.listdir(dir_path) if item.endswith(".lab")]
    res = {}
    for label_name in label_names:
        # Loop all label file
        path = os.path.join(dir_path, label_name)
        with open(path, "r") as f:
            lines = f.read().splitlines()
            name = label_name[:label_name.rfind(".")]
            # save f0
            res[name] = {
                "mean": int(lines[-2].split()[-1]),
                "std": int(lines[-1].split()[-1]),
            }
            # save time label
            for line in lines[:-2]:
                line = line.split()
                start = float(line[0])
                end = float(line[1])
                label = line[2]
                if label not in res[name]:
                    res[name][label] = []
                res[name][label].append((start, end))
    return res


def get_label_of_time(label, time):
    """
    Find the lable of specical time
    :param label: dictionary contain label of curent audio
    :param time: time which need to get lable
    :return: string of v or uv or sil as the label
    """
    for item in label["sil"]:
        if item[0] <= time <= item[1]:
            return "sil"

    for item in label["v"]:
        if item[0] <= time <= item[1]:
            return "v"

    for item in label["uv"]:
        if item[0] <= time <= item[1]:
            return "uv"
    return None
