import json

def txt_2_Json(txt : str) -> dict[str, any]:
    lineas = [linea.strip() for linea in txt.strip().split("\n") if linea.strip()]
    pares = [linea.split(": ", 1) for linea in lineas]
    datos = {clave.strip(): valor.strip() for clave, valor in pares}
    res = json.dumps(datos, indent=4)
    res = json.loads(res)
    
    return res