import enum


class NetworkOptimizer(enum.Enum):
    ADAM = 1,
    ADAMAX = 2,
    RMSPROP = 3,
    SGD = 4,
    ADAGRAD = 5,

    @staticmethod
    def all():
        return [NetworkOptimizer.ADAM, NetworkOptimizer.ADAMAX, NetworkOptimizer.RMSPROP, NetworkOptimizer.SGD,
                NetworkOptimizer.ADAGRAD]

    @staticmethod
    def get_type_by_name(name):
        for optimizer_type in NetworkOptimizer.all():
            if optimizer_type.name.lower() == name.lower():
                return optimizer_type
        return None


class PolicyType(enum.Enum):
    EPSILON_GREEDY = 0,
    SOFTMAX = 1,
    THOMPSON_SAMPLING = 2,
    UCB = 3

    @staticmethod
    def all():
        return [PolicyType.EPSILON_GREEDY, PolicyType.SOFTMAX, PolicyType.THOMPSON_SAMPLING, PolicyType.UCB]


class TDAlgorithmType(enum.Enum):
    SARSA = 0
    Q = 1
    EXPECTED_SARSA = 2
    MCQ = 3

    @staticmethod
    def all():
        return [TDAlgorithmType.SARSA, TDAlgorithmType.Q, TDAlgorithmType.EXPECTED_SARSA, TDAlgorithmType.MCQ]

    @staticmethod
    def get_type_by_name(name):
        for algorithm_type in TDAlgorithmType.all():
            if algorithm_type.name.lower() == name.lower():
                return algorithm_type
        return None


class LearningType(enum.Enum):
    OFFLINE = 0,
    ONLINE = 1,
    BOTH = 2
