import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    #  load in all directories from APSO directory
    APSO_dir = os.path.join(os.getcwd(), 'APSO')
    APSO_dirs = [os.path.join(APSO_dir, d) for d in os.listdir(APSO_dir) if os.path.isdir(os.path.join(APSO_dir, d))]

    for APSO_dir in APSO_dirs:
        # get the csv file
        csv_file = [os.path.join(APSO_dir, f) for f in os.listdir(APSO_dir) if f.endswith('.csv')][0]
        print(csv_file)
        diff = science(csv_file)
        #  make a new directory for the diff
        os.makedirs(os.path.join(APSO_dir, 'diff'), exist_ok=True)
        for index in range(diff.shape[0]):
            plt.imshow(diff[index].reshape(28,28), cmap='gray')
            plt.savefig(os.path.join(APSO_dir, 'diff', f'{index}.png'))
            plt.close()
            plt.clf()
            



def science(csv_file):
    data = pd.read_csv(csv_file)

    #  convert the string representation of the array to an array
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            string_representation = data.iloc[row, col]
            cleaned_string = string_representation.replace('[', '').replace(']', '').replace('\n', '').replace('  ', ' ')
            values = cleaned_string.split()
            array = np.array(values, dtype=np.float32)
            #  replace the string with the array
            data.iloc[row, col] = array

    #  get the initial and final values
    initial = data.iloc[0, :]
    final = data.iloc[-1, :]

    #  get the difference
    diff = np.subtract(final , initial)
    return diff

    



if __name__ == '__main__':
    main()
