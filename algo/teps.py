from collections import deque


class TEPS:
    """
    Tool Engagement based Phase Segmentation (TEPS) system.
    
    Dynamically classifies signal phases (e.g., drilling vs. air) using 
    adaptive clustering and rolling window statistics.
    """

    def __init__(
        self,
        start_mean_air=(0.0, 0.0, 0.0),
        start_mean_drill=(-1.5, -1.5, -1.5),
        ignore_start=30,
        init_mode='fixed',
        min_distance_threshold=0.1,
        hist_size=20,
        weights=(1.0, 1.0, 1.0),
        factor=1,
        a_init=0.01,
        a_min=0.001,
        decay_factor=0.9995
    ):
        """
        Initialize TEPS with cluster defaults, decay parameters, and configuration.
        """
        # Modes: "fixed", "min_distance"
        self.air_cluster = {
            'min': start_mean_air[0],
            'max': start_mean_air[1],
            'mean': start_mean_air[2],
            'count': 0,
            'sum': 0,
            'ready': False
        }
        self.drill_cluster = {
            'min': start_mean_drill[0],
            'max': start_mean_drill[1],
            'mean': start_mean_drill[2],
            'count': 0,
            'sum': 0,
            'ready': False
        }

        # Init step 
        self.a_init = a_init  # Starting learning rate
        self.a_min = a_min  # Final learning rate
        self.a = self.a_init
        self.decay_factor = decay_factor

        self.start_index = 0
        self.ignore_start = ignore_start
        self.init_mode = init_mode
        self.min_distance_threshold = min_distance_threshold
        self.hist_size = hist_size
        self.x_hist = deque(maxlen=hist_size)  # Rolling buffer of input samples
        self.weights = weights  # Weights for mean, min, max in distance
        self.factor = factor  # Controls torque direction polarity

    def process_sample(self, x):
        """
        Process a new sample and return prediction and current cluster means.
        
        Returns:
            tuple: (prediction, (mean_air, mean_drill))
        """
        if self.ignore_start > self.start_index:
            self.start_index += 1
        else:
            self.x_hist.append(x)
            self.update()
            self.update_step_size()

        p = self.predict()
        return p, (self.air_cluster['mean'], self.drill_cluster['mean'])

    def update_step_size(self):
        """
        Decay the internal learning rate `a` over time until it reaches `a_min`.
        """
        if self.a > self.a_min:
            self.a *= self.decay_factor
            if self.a < self.a_min:
                self.a = self.a_min

    def predict(self):
        """
        Predict if the signal corresponds to 'drill engaged' (0) or 'drill not engaged' (1).
        """
        if len(self.x_hist) < self.hist_size:
            return 0  # Not enough samples yet

        x_min, x_max, x_mean = self.compute_rolling_stats()
        air_dist = self.compute_cluster_distance(self.air_cluster, x_min, x_max, x_mean)
        drill_cluster = self.get_drill_or_clipped_cluster()
        drill_dist = self.compute_cluster_distance(drill_cluster, x_min, x_max, x_mean)

        # Predict air if clusters are too close
        if (
            self.init_mode == 'min_distance'
            and abs(self.drill_cluster['mean'] - self.air_cluster['mean']) < self.min_distance_threshold
        ):
            return 0

        return 0 if drill_dist > air_dist else 1

    def get_drill_or_clipped_cluster(self):
        """
        Get the drilled cluster, or a clipped cluster. Usefull if the drill signal changes a lot over time.
        """
        shift = 2 * abs(self.air_cluster['min'] - self.air_cluster['max'])
        clipped_mean = self.air_cluster['mean'] + self.factor * shift
        clipped_max = self.air_cluster['max'] + self.factor * shift
        clipped_min = self.air_cluster['min'] + self.factor * shift

        if (
            abs(clipped_min - self.air_cluster['min']) < abs(clipped_min - self.drill_cluster['min'])
            and abs(clipped_max - self.air_cluster['max']) < abs(clipped_max - self.drill_cluster['max'])
        ):
            return {'mean': clipped_mean, 'max': clipped_max, 'min': clipped_min}
        else:
            return self.drill_cluster

    def compute_rolling_stats(self):
        """
        Compute min, max, and mean from the rolling sample buffer.
        
        Returns:
            tuple: (x_min, x_max, x_mean)
        """
        if len(self.x_hist) == 0:
            return None, None, None
        return min(self.x_hist), max(self.x_hist), sum(self.x_hist) / len(self.x_hist)

    def compute_cluster_distance(self, cluster, x_min, x_max, x_mean):
        """
        Compute a weighted distance from a cluster to a sample window.
        """
        w1, w2, w3 = self.weights
        mean_dist = abs(x_mean - cluster['mean'])
        min_dist = abs(x_min - cluster['min'])
        max_dist = abs(x_max - cluster['max'])
        return (w1 * mean_dist + w2 * min_dist + w3 * max_dist) / (w1 + w2 + w3)

    def update(self):
        """
        Update the cluster closest to the current window.
        """
        if len(self.x_hist) < self.hist_size:
            return

        x_min, x_max, x_mean = self.compute_rolling_stats()

        # Init clusters
        if self.init_mode == 'min_distance' and not self.air_cluster['ready']:
            self.air_cluster['mean'] = x_mean
            self.air_cluster['min'] = x_min
            self.air_cluster['max'] = x_max
            self.air_cluster['ready'] = True
            self.drill_cluster['mean'] = x_mean + self.factor * abs(self.min_distance_threshold / 2)
            self.drill_cluster['min'] = x_min + self.factor * abs(self.min_distance_threshold / 2)
            self.drill_cluster['max'] = x_max + self.factor * abs(self.min_distance_threshold / 2)
            return

        air_dist = self.compute_cluster_distance(self.air_cluster, x_min, x_max, x_mean)
        drill_dist = self.compute_cluster_distance(self.drill_cluster, x_min, x_max, x_mean)

        clstr, _ = (self.air_cluster, self.drill_cluster) if air_dist <= drill_dist else (self.drill_cluster, self.air_cluster)

        clstr['mean'] = (1 - self.a) * clstr['mean'] + self.a * x_mean
        clstr['min'] = (1 - self.a) * clstr['min'] + self.a * x_min
        clstr['max'] = (1 - self.a) * clstr['max'] + self.a * x_max

    def get_phase_label(self, last_state, treshold, feed):
        """
        Map torque state + feed direction + previous phase into a discrete phase label.

        0 - rapid traverse
        1 - air drilling
        2 - drilling
        3 - unexp. drop
        4 - exp. drop
        5 - repositioning

        Returns:
            int: phase label [0–5]
        """

        if abs(feed) > 1000:
            return 0

        if not last_state:
            if treshold == 1 and feed < 0:  # starting with drilling
                return 2
            elif treshold == 1:  # feed is not moving, or moving up + torque
                return 4
            elif feed >= 0:  # resting / moving up
                return 5
            else:  # moving down and no torque
                return 1

        if last_state == 1:
            if treshold < 1 and feed < 0:
                return 1
            elif treshold < 1 and feed >= 0:
                return 5
            else:
                return 2

        if last_state == 2 or last_state == 3:
            if treshold == 1 and feed < 0:
                return 2
            elif treshold < 1 and feed < 0:
                return 3
            elif feed >= 0 and treshold == 1:
                return 4
            else:
                return 5

        if last_state == 4 or last_state == 5:
            if feed >= 0 and treshold < 1:
                return 5
            elif feed >= 0 and treshold == 1:
                return 4
            else:
                return 1
