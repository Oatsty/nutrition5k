from typing import Any


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val: int | float, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeterDict(object):
    def __init__(self, keys):
        self.keys = keys
        self.average_meter_dict = {key: AverageMeter() for key in keys}

    def reset(self):
        for key in self.keys:
            self.average_meter_dict[key].reset()

    def update(self, val: dict[str, Any], n=1):
        for key in val.keys():
            if key not in self.keys:
                raise ValueError(f"{self} does not have key {key}")
            self.average_meter_dict[key].update(val[key], n)

    def get_sum(self, key):
        if key not in self.keys:
            raise ValueError(f"{self} does not have key {key}")
        return self.average_meter_dict[key].sum

    def get_avg(self, key):
        if key not in self.keys:
            raise ValueError(f"{self} does not have key {key}")
        return self.average_meter_dict[key].avg

    def iter_sum(self):
        for key in self.keys:
            yield key, self.average_meter_dict[key].sum

    def iter_avg(self):
        for key in self.keys:
            yield key, self.average_meter_dict[key].avg

    def get_keys(self):
        return self.average_meter_dict.keys()
