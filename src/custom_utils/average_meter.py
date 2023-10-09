from typing import Any


class AverageMeter(object):
    """
    Average Meter for recording running value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Reset value to 0
        """
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val: int | float, n=1):
        """
        Update current value, sum, count, and average value.
        A value can be count multiple times (e.g., batches with different sizes).

        Args:
            val (int | float): value to be update
            n (int): number of times to be couted

        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeterDict(object):
    """
    Dictionary of Average Meters

    Args:
        keys (Iterable): List of keys for each Average Meter
    """
    def __init__(self, keys):
        self.keys = keys
        self.average_meter_dict = {key: AverageMeter() for key in keys}

    def reset(self):
        """
        Reset all Average Meters
        """
        for key in self.keys:
            self.average_meter_dict[key].reset()

    def update(self, val: dict[str, Any], n=1):
        """
        Update all Average Meters

        Args:
            val (dict[str, int | float]): dictionary of values for Average Meters that need to be updated.
            keys need to exist in self.keys
            n (int): number of times to be couted
        """
        for key in val.keys():
            if key not in self.keys:
                raise ValueError(f"{self} does not have key {key}")
            self.average_meter_dict[key].update(val[key], n)

    def get_sum(self, key):
        """
        Get the summation of a specific Average Meter

        Args:
            key (str): an average meter key
        Return:
            sum (int | float): summation of the value in the specified Average Meter
        """
        if key not in self.keys:
            raise ValueError(f"{self} does not have key {key}")
        return self.average_meter_dict[key].sum

    def get_avg(self, key):
        """
        Get the average of a specific Average Meter

        Args:
            key (str): an average meter key
        Return:
            int | float: Average of the value in the specified Average Meter
        """
        if key not in self.keys:
            raise ValueError(f"{self} does not have key {key}")
        return self.average_meter_dict[key].avg

    def iter_sum(self):
        """
        Get the summations of all Average Meters

        Args:
            key (str): an average meter key
        Return:
            Iterable[tuple[key, val]]: Keys and corresponding summations of the values in all Average Meters
        """
        for key in self.keys:
            yield key, self.average_meter_dict[key].sum

    def iter_avg(self):
        """
        Get the averages of all Average Meters

        Args:
            key (str): an average meter key
        Return:
            Iterable[tuple[key, val]]: Keys and corresponding averages of the values in all Average Meters
        """
        for key in self.keys:
            yield key, self.average_meter_dict[key].avg

    def get_keys(self):
        """
        Get self.keys

        Args:
            key (str): an average meter key
        Return:
            keys: self.keys
        """
        return self.average_meter_dict.keys()
