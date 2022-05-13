import pickle
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collision_file", type=str, default="collision_1")
    parser.add_argument("--fail_file", type=str, default="fail_1")
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.1)
    args = parser.parse_args()

    with (open(args.collision_file, "rb")) as openfile:
        collisions = pickle.load(openfile)
    collisions = np.array(collisions)
    safe_rate = 1 - np.sum(collisions) / len(collisions)
    print("Empirical safe rate", safe_rate)

    # with (open(args.fail_file, "rb")) as openfile:
    #     fails = pickle.load(openfile)
    # fails = np.array(fails)
    # safe_rate = safe_rate * (1 - np.sum(fails) / len(fails))

    epsilon = args.epsilon
    delta = args.delta
    N = collisions.shape[0]
    true_safey_rate = safe_rate - np.sqrt(np.log(1/delta)/2/N)
    print('True safe rate is no less than:', true_safey_rate)
    if true_safey_rate > 1 - args.epsilon:
        print("Safe")
    else:
        print("Not safe")