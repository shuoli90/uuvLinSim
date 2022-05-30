import pickle
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters_file", type=str, default="parameters_2.pickle")
    args = parser.parse_args()

    with open(args.parameters_file, 'rb') as handle:
        parameters = pickle.load(handle)
    
    with open(args.parameters_file, 'wb') as handle:
        pickle.dump(parameters, handle, protocol=4)

    print(parameters)