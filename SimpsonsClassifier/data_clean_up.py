# Дополнительная функция, удаляющая редкие классы, если требуется
def data_class_clean_up():
    import os
    import pandas as pd
    import shutil

    characters = os.listdir("journey-springfield/train/train")
    print(characters)
    df_characters_found = pd.read_csv("journey-springfield/submission_final.csv")
    characters_found = df_characters_found["Expected"].unique()
    print(len(characters_found))
    for character in characters:
        if character not in characters_found:
            if os.path.isdir('journey-springfield/train/train/' + character):
                shutil.rmtree('journey-springfield/train/train/' + character)
