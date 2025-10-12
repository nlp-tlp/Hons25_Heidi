import pandas as pd

# Fill downward for specified columns
def clean_csv(input_file, output_file, fill_columns):
    df = pd.read_csv(input_file)

    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x) # Strip whitespace
    df[fill_columns] = df[fill_columns].fillna(method='ffill') # Fill blanks downward for specified columns

    df.to_csv(output_file, index=False)

fill_columns = ["Subsystem", "Component", "Sub-Component", "Potential Failure Mode", "Potential Effect(s) of Failure", "Potential Cause(s) of Failure"] # note that not all columns are filled - see "Recommended action"
clean_csv("fmea_dataset_combined.csv", "fmea_dataset_filled.csv", fill_columns)
