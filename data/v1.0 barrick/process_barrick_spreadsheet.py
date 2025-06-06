import pandas as pd
import json

"""
Fill downward for specified columns
"""
def clean_csv(input_file, output_file, fill_columns):
    df = pd.read_csv(input_file)

    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x) # Strip whitespace
    df[fill_columns] = df[fill_columns].fillna(method='ffill') # Fill blanks downward for specified columns

    df.to_csv(output_file, index=False)

fill_columns = ["Spreadsheet", "Subsystem", "Component", "Sub-Component", "Potential Failure Mode", "Potential Effect(s) of Failure", "Potential Cause(s) of Failure"] # note that not all columns are filled - see "Recommended action"
clean_csv("fmea_barrick_original.csv", "fmea_barrick_filled.csv", fill_columns)


"""
Merge all fields into one column for use in GraphRAG, making text as the column name
Because GraphRAG can't actually deal with multiple CSV columns :(
"""
def merge_csv(input_file, output_file):
    df = pd.read_csv(input_file)

    df["text"] = df.apply(lambda row: ", ".join(f"{col}: {val}" for col, val in row.dropna().items()), axis=1)
    df = df[["text"]]

    df.to_csv(output_file, index=False)

merge_csv("fmea_barrick_filled.csv", "fmea_barrick_merged.csv")
