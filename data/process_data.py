import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Loads the data specified by the paths in the arguments and merges it to one dataframe.

    Args:
        messages_filepath: The file path to the messages csv. Must contain an 'id' column.
        categories_filepath: The file path to the categories csv. Must contain an 'id' column.

    Returns:
        Combined dataframe holding both files.

    """
    # Load files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge files
    df = pd.merge(messages, categories, on='id', how='inner')

    return df


def clean_data(df):
    """Cleans the dataframe by expanding columns which were initially concatenated. Also removes duplicate rows.

    Args:
        df: The dataframe to be cleaned.

    Returns:
        Cleaned dataframe

    """
    # Expand concatenated categories into new dataframe
    categories = df.categories.str.split(';', expand=True)

    # Extract column names and set columns in new dataframe
    row = categories.head(1)
    category_colnames = list(map(lambda x: x.split('-')[0], row.values.tolist()[0]))
    categories.columns = category_colnames

    # Convert column values to actual integer values (before: colA-1 --> after: 1)
    for column in categories:
        categories[column] = categories[column].map(lambda x: x.split('-')[1])
        categories[column] = categories[column].map(lambda x: int(x))

    # Replace existing categories column with expanded columns
    df.drop(columns=["categories"], inplace = True)
    df = pd.concat([df, categories], axis=1)

    # Drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """Stores the dataframe as a sqlite database.

    Args:
        df: The dataframe to be stored.
        database_filename: The database filename under which it will be stored. Must be in the format of "database.db".
            Will also be used for the table name (without file extension)

    Returns:
        Nothing

    """
    # Store dataframe to sqlite db
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('{}'.format(database_filename[database_filename.rindex('/')+1:].replace('.db','')), engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()