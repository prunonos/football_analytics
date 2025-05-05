import argparse
from collections import OrderedDict
import os
import re
import yaml

def generar_nombre_archivo(data):
    experiment = data["experiment"]
    balanced = "_bal" if data["data"].get("balance",False) else ""
    sampling = "_ss" if  data["data"].get("sample")=="sequential" else "_rs"
    batch = "_batch" if data["data"].get("select",False) else ""
    advanced = '_adv' if data["data"].get("features",False) and 'top1' in experiment else ""
    rps_arg = data["training"].get("eval_metric",'')
    rps = '_rps' if rps_arg=='rps' or ('rps' in rps_arg) else "" 
    supervised, aggregations, factor = "", "", ""
    if experiment.startswith("tabnet"):
        supervised = "_sup" if data["training"]["unsupervised_training"] else "_unsup"
        if experiment.endswith("own") or experiment.endswith("owntop1"):
            aggregations = "_comp" if "S" in data["data"]["aggregations"].keys() else "_simp"
    elif experiment.startswith("own"):
        aggregations = "_comp" if "S" in data["data"]["aggregations"] else "_simp"
        factor = "_fac" if  data["data"].get("factor",False) else "_nofac"
    else:
        name = data["experiment_id"]
    
    name = experiment + advanced + aggregations + balanced + factor + sampling + supervised + rps + batch
    data["experiment_id"] = name
    print(f"New name: {name}\n")
    return name + '.yaml'

def modificar_yaml(archivos, campos_a_modificar, replace=True, drop=[], adding=[]):
    for archivo in archivos:
        # Cargar el archivo YAML
        with open(archivo, 'r') as file:
            data = yaml.safe_load(file)

        # Crear un OrderedDict para mantener el orden de los campos
        nuevo_data = OrderedDict()

        # Copiamos data en el OrderedDict
        for campo, valor in data.items():
            nuevo_data[campo] = valor

        for d in drop:
            nuevo_data = remove_key_path(nuevo_data,d)

        # Modificar campos segÃºn los valores proporcionados
        for campo, valor in campos_a_modificar.items():
            # Usar un valor predeterminado si el campo no estÃ¡ presente en el YAML original
            if isinstance(nuevo_data.get(campo,None),dict):
                nuevo_data[campo].update(valor)
            elif isinstance(nuevo_data.get(campo,None),list) and (campo in adding):
                nuevo_data[campo] += valor
            else:
                nuevo_data[campo] = valor
            print(archivo,campo,valor)

        name = generar_nombre_archivo(nuevo_data)

        # Guardar el archivo modificado
        with open(name, 'w') as filew:
            yaml.dump(dict(nuevo_data), filew, default_flow_style=False)
        
        if replace and name!=archivo:
            os.remove(archivo)

    return name
   
def parse_elements(string):
    pattern = r'\[([^\[\]]*)\]'
    match = re.search(pattern, string)
    if match:
        elements_str = match.group(1)
        elements = [elem.strip() for elem in elements_str.split(',')]
        parsed_elements = []
        for elem in elements:
            if elem.isdigit():
                parsed_elem = int(elem)
            else:
                try:
                    parsed_elem = float(elem)
                except ValueError:
                    parsed_elem = elem
            parsed_elements.append(parsed_elem)
        return parsed_elements
    else:
        return string

def update_nested_dict(d, keys, value):
    if keys=='': return d
    key_list = keys.split(".")
    current_dict = d
    for key in key_list[:-1]:
        current_dict = current_dict.setdefault(key, {})
    if value.isdigit(): value = int(value)
    elif value.isdecimal(): value = float(value)
    value = parse_elements(value) if isinstance(value,str) else value
    current_dict[key_list[-1]] = value
    return d

def remove_key_path(d, keys):
    key_list = keys.split(".")
    current_dict = d
    for key in key_list[:-1]:
        current_dict = current_dict.get(key, {})
    
    if key_list[-1] in current_dict:
        del current_dict[key_list[-1]]
    return d


if __name__ == "__main__":
    # Lista de archivos YAML a modificar
    parser = argparse.ArgumentParser(description='Modificar archivos YAML.')
    parser.add_argument('-r','--replace',action="store_true",default=False)
    parser.add_argument('archivos', nargs='+', help='Archivos YAML a modificar')
    parser.add_argument('-d','--drop', type=str, nargs='*',default=[]) # si es un campo anidado tal que asi "A" : { "B": 1 } -> A.B
    parser.add_argument('-a','--add', type=str, nargs='*',default=[])  # si es un campo tipo lista se añade el valor a la lista
    parser.add_argument('-f','--fields',nargs='*',default=[])
    parser.add_argument('-v','--values',nargs='*',default=[])
    args = parser.parse_args()

    # Campos y valores a modificar
    campos_modificar = {
        "enable":True,
        # "data": {"select":1000},
        # "training": {
        #     "select": 1000,
        #     "timeout":3700,
        #     "max_epochs": 120,
        #     "iterations": 5000
        # }
    }

    keys = args.fields
    values = args.values
    for key,value in zip(keys,values):
        campos_modificar = update_nested_dict(campos_modificar,key,value)
    print(campos_modificar)
    # Modificar archivos YAML
    modificar_yaml(args.archivos, campos_modificar, replace=args.replace, drop=args.drop, adding=args.add)
