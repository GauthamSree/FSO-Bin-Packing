import numpy as np
import copy
import operator, functools

from scipy.stats.stats import rankdata

class BinDetails:
    def __init__(self, bins, free_bin_caps, rov_continous):
        self.bins = bins
        self.free_bin_caps = free_bin_caps
        self.no_of_bins = len(bins)
        self.rov_continous = rov_continous
        self.score = -1
    
    def __repr__(self):
        return f"BinDetails (Score = {self.score}, No_of_bins = {self.no_of_bins})"
    
    def __str__(self):
        return f"BinDetails (Score = {self.score}, No_of_bins = {self.no_of_bins})"

class BinPackingSolutions:
    def __init__(self):
        self.bin_solutions: dict[int, BinDetails] = {}
        self.max_bin_capacity = 30
        self.items = np.array([5, 14, 11, 8, 3, 2, 13, 1, 5, 6, 9, 4, 7, 12, 10, 15])
        self.no_of_items = self.items.size
        self.rng = np.random.default_rng(12345)

    def __repr__(self):
        return f"BinPackingSolutions (No_of_items = {self.no_of_items}, Max_bin_capacity = {self.max_bin_capacity})"
    
    def __str__(self):
        return f"BinPackingSolutions (No_of_items = {self.no_of_items}, Max_bin_capacity = {self.max_bin_capacity})"

    def extract_from_file(self, file_path):
        with open(file_path, "r") as f: 
            lines = f.readlines()
            n = int(lines[0])
            capacity = int(lines[1])
            items = [int(lines[i]) for i in range(2, n+2)]
        
        self.items = np.array(items)
        self.no_of_items = n
        self.max_bin_capacity = capacity

    def best_fit_algorithm(self, ranked_values):
        free_bin_caps = [self.max_bin_capacity]
        bins = [[]]
        rov_continous = [[]]
        idxs = rankdata(ranked_values, method='ordinal') - 1
        
        ranked_items = copy.deepcopy(self.items)  
        ranked_items = ranked_items[idxs]

        for i in range(self.no_of_items):
            item = ranked_items[i]
            ranked_val = ranked_values[i]
            free_caps = np.array(free_bin_caps)
            new_caps = (free_caps - item)
            valid_idx = np.where(new_caps >= 0)[0]
            if valid_idx.size:
                idx = valid_idx[new_caps[valid_idx].argmin()]
                bins[idx].append(item)
                rov_continous[idx].append(ranked_val)
                free_bin_caps[idx] -= item
            else:
                bins.append([item])
                rov_continous.append([ranked_val])
                free_bin_caps.append(self.max_bin_capacity - item)
        
        rov_continous = np.array(functools.reduce(operator.iconcat, rov_continous, []))
        new_bin = BinDetails(bins, free_bin_caps, rov_continous)
        return new_bin

    def best_fit_heuristic(self):
        free_bin_caps = [self.max_bin_capacity]
        bins = [[]]
        rov_continous = [[]]
        ranked_values = self.rng.uniform(0, 5, self.items.shape[0])
        idxs = rankdata(ranked_values, method='ordinal') - 1
        
        ranked_items = copy.deepcopy(self.items)  
        ranked_items = ranked_items[idxs]

        for i in range(self.no_of_items):
            item = ranked_items[i]
            ranked_val = ranked_values[i]
            free_caps = np.array(free_bin_caps)
            new_caps = (free_caps - item)
            valid_idx = np.where(new_caps >= 0)[0]
            if valid_idx.size:
                idx = valid_idx[new_caps[valid_idx].argmin()]
                bins[idx].append(item)
                rov_continous[idx].append(ranked_val)
                free_bin_caps[idx] -= item
            else:
                bins.append([item])
                rov_continous.append([ranked_val])
                free_bin_caps.append(self.max_bin_capacity - item)
        
        rov_continous = np.array(functools.reduce(operator.iconcat, rov_continous, []))
        new_bin = BinDetails(bins, free_bin_caps, rov_continous)
        return new_bin


class FSODetails:
    def __init__(self, female, female_score):
        self.female = female
        self.female_score = female_score
        self.best_score_id = np.argmin(female_score)
        self.x = female[self.best_score_id]
    
    def __repr__(self):
        return f"FSODetails (X = {self.x}, Best_Score = {self.female_score[self.best_score_id]})"
    
    def __str__(self):
        return f"FSODetails (X = {self.x}, Best_Score = {self.female_score[self.best_score_id]})"