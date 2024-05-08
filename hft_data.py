import numpy as np
import pandas
import argparse

"""
process HFT data
"""


# ---- OLD STUFF DELETE LATER ----- #
def getDtype(filename):
    with open(filename, "r") as f:
        labels = f.readline().strip().split(';')  # first row of csv is headers
        types  = []
        firstColumn = True
        for column in f.readline().strip().split(';'):
            if firstColumn:
                if labels[0] == "fDetId":
                    types.append(np.dtype(np.int8))
                firstColumn = False
            if '.' in column:
                types.append(np.float32)
            else:
                types.append(np.int32)
    lst = [("event",np.dtype(np.int16))]
    for i in range(len(labels)):
        field = (labels[i], types[i])
        lst.append(field)
    dtype = np.dtype(lst)
    return dtype

def npzFromCSV(csv_paths, name, dtype):
    """ Makes name.npz file with dtype from csv_paths """
    tracks = False
    with tqdm(total = len(csv_paths)) as pbar:
        for csv_path in csv_paths: 
            event = int(csv_path.split('_')[-2][5:])
            data = np.loadtxt(csv_path, delimiter=';', skiprows=1)
            if data.size:  # data may be empty - no tracks
                if len(data.shape) == 1: # need to make 2D at least
                    data = np.array([data])
                    print(data)
                event_column = np.atleast_2d(np.tile(event,data.shape[0])).T
                data_w_event = rf.unstructured_to_structured(np.hstack((event_column, data)),dtype)
                if isinstance(tracks, bool):
                    tracks = np.array(data_w_event, dtype)
                else:
                    tracks = np.append(tracks, data_w_event, axis=0)
            pbar.update(1)
    print(tracks.shape)
    print(tracks[0])
    print(dtype)
    np.savez_compressed(name+'.npz', tracks)
# ---- OLD STUFF DELETE LATER ----- #


def main(args):
    print(args)
    print("yoyoyo")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", dest="maketracks", default=False, action='store_true',
                        help="Make tracks.npz")
    parser.add_argument("--clusters", dest="makeclusters", default=False, action='store_true',
                        help="Make clusters.npz")
    parser.add_argument("--nodtype", dest="showdtype", default=True, action='store_false', help="Show dtype of both")
    parser.add_argument("--csvs", dest="makecsvs", default=False, action='store_true', help="makecsvs")
    args = parser.parse_args()
    main(args)
