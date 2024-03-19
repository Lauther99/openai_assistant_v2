import pandas as pd
from typing import List, Dict


def read_excel_dictionary(excel_path: str, columnas: List[str]) -> Dict[str, List[str]]:
    dataframe = pd.read_excel(excel_path, usecols=columnas)
    dataframe_sin_nan = dataframe.dropna()

    data = []

    # Itera sobre cada fila del DataFrame
    for indice, fila in dataframe_sin_nan.iterrows():
        datos_fila = {
            "table_name": fila["Tabla"],
            "keywords": fila["Palabras clave"].split(","),
        }
        data.append(datos_fila)

    return data


def read_keywords_dictionary(
    excel_path: str, columnas: List[str]
) -> Dict[str, List[str]]:
    dataframe = pd.read_excel(excel_path, usecols=columnas)
    dataframe_sin_nan = dataframe.dropna()

    data = []

    # Itera sobre cada fila del DataFrame
    for indice, fila in dataframe_sin_nan.iterrows():
        datos_fila = {"request": fila["Request"], "keywords": fila["Keywords"]}
        data.append(datos_fila)

    return data


def read_summaries_dictionary(
    excel_path: str, columnas: List[str]
) -> Dict[str, List[str]]:
    dataframe = pd.read_excel(excel_path, usecols=columnas)
    dataframe_sin_nan = dataframe.dropna()

    data = []

    # Itera sobre cada fila del DataFrame
    for indice, fila in dataframe_sin_nan.iterrows():
        datos_fila = {"summary": fila["summary"], "request": fila["request"]}
        data.append(datos_fila)

    return data


def read_classify_dictionary(
    excel_path: str, columnas: List[str]
) -> Dict[str, List[str]]:
    dataframe = pd.read_excel(excel_path, usecols=columnas)
    dataframe_sin_nan = dataframe.dropna()

    data = []

    # Itera sobre cada fila del DataFrame
    for indice, fila in dataframe_sin_nan.iterrows():
        datos_fila = {
            "input": fila["input"],
            "analysis": fila["analysis"],
            "response": fila["response"],
        }
        data.append(datos_fila)

    return data


def get_excel_data(
    excel_path: str, columnas: List[str], sheet_name: str = "Hoja1"
) :
    dataframe = pd.read_excel(excel_path, usecols=columnas, sheet_name=sheet_name)
    
    ddl_arr, variables_arr, examples_arr, db_info_arr = [], [], [], []
    # Itera sobre cada fila del DataFrame
    for _, fila in dataframe.iterrows():
        if not pd.isna(fila["ddl"]):
            ddl_arr.append(fila["ddl"])
            
        if not pd.isna(fila["about_variables"]):
            variables_arr.append(fila["about_variables"])
            
        if not pd.isna(fila["about_database"]):
            db_info_arr.append(fila["about_database"])
                        
        if not pd.isna(fila["questions"]) and not pd.isna(fila["answers"]):
            example = {
                "question": fila["questions"],
                "answer": fila["answers"]
            }
            examples_arr.append(example)
        
    data = {
        "ddl" : ddl_arr,
        "about_variables" : variables_arr,
        "about_database" : db_info_arr,
        "examples" : examples_arr,
    }

    return data