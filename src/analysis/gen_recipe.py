import os, sys
current_dir = os.path.split(os.path.abspath(__file__))[0]
proj_dir = current_dir.rsplit('/', 2)[0]
sys.path.append(proj_dir)

import json
from tqdm import tqdm 
import matplotlib.pyplot as plt
import pandas as pd

def list_direct_subdirectories(directory):
    entries = os.listdir(directory)
    subdirectories = [os.path.join(directory, entry) for entry in entries if os.path.isdir(os.path.join(directory, entry))]
    return subdirectories

def plot_logic(folder_path,logics):
    df = pd.DataFrame()

    for logic in logics:
        subfolder_path = os.path.join(folder_path, logic)
        for file_name in os.listdir(subfolder_path):
            if file_name.startswith('recipe') and file_name.endswith('.logic.qor.json'):
                file_path = os.path.join(subfolder_path, file_name)
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    # Extract the area and delay
                    area = data['area']
                    delay = data['delay']
                    # Create a temporary DataFrame to store the data for the current file
                    temp_df = pd.DataFrame({
                        'logic': [logic],
                        'area': [area],
                        'delay': [delay],
                        'file': [file_name]
                    })
                    df = pd.concat([df, temp_df], ignore_index=True)

    # Extract the x number from the file name
    df['x'] = df['file'].apply(lambda x: x.split('_')[1])

    unique_x = df['x'].unique()

    # Plot a scatter plot for each recipe
    for x in tqdm(unique_x, desc='Processing recipes'):
        fig, ax = plt.subplots()
        subset = df[df['x'] == x]
        for logic in logics:
            logic_data = subset[subset['logic'] == logic]
            ax.scatter(logic_data['area'], logic_data['delay'], label=logic)
            # Add text labels around each point
            for index, row in logic_data.iterrows():
                ax.annotate(logic, (row['area'], row['delay']), textcoords="offset points", xytext=(0,10), ha='center')

        ax.set_xlabel('Area')
        ax.set_ylabel('Delay')
        ax.legend()
        plt.title(f'Area vs Delay for Recipe {x}')

        pdf_path = os.path.join(folder_path, f'qor_{x}_logics.pdf')
        plt.savefig(pdf_path, format='pdf')
        plt.close(fig)  

if __name__ == '__main__':
    logics = ["abc", "aig", "oig", "aog", "xag", "xog", "primary", "mig", "xmg", "gtg"]
    folder = sys.argv[1]
    folders_dir = list_direct_subdirectories(folder)
    for folder_dir in folders_dir:
        base_name = os.path.basename(folder_dir)
        name = base_name.split('_')[0]
        print(name)
        plot_logic(folder_dir,logics)