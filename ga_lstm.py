import tensorflow as tf

from data_process import csv_to_dataset


def main():

    X, Y = csv_to_dataset(r'Metro_Interstate_Traffic_Volume.csv')

    pass



if __name__ == "__main__":
    main()
