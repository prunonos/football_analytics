import argparse
from collections import OrderedDict
import yaml

def generar_nombre_archivo(data):
    # Obtener substrings según reglas
    experiment = data["experiment"]
    balanced = "_bal" if data["data"].get("balance",False) else ""
    sampling = "_ss" if  data["data"].get("sample")=="sequential" else "_rs"
    if experiment.startswith("tabnet"):
        supervised = "_sup" if data["training"]["unsupervised_training"] else "_unsup"
        if experiment.endswith("own"):
            aggregations = "_comp" if "S" in data["data"]["aggregations"].keys() else "_simp"
        factor = ""
    else:
        supervised = ""
        aggregations = "_comp" if "S" in data["data"]["aggregations"] else "_simp"
        factor = "_fac" if  data["data"].get("factor",False) else "_nofac"
    name = experiment + aggregations + balanced + factor + sampling + supervised
    data["experiment_id"] = name
    print(f"New name: {name}\n")
    return name + '.yaml'

def modificar_yaml(archivos, campos_a_modificar):
    for archivo in archivos:
        # Cargar el archivo YAML
        with open(archivo, 'r') as file:
            data = yaml.safe_load(file)

        # Crear un OrderedDict para mantener el orden de los campos
        nuevo_data = OrderedDict()

        # Copiamos data en el OrderedDict
        for campo, valor in data.items():
            nuevo_data[campo] = valor

        # Modificar campos según los valores proporcionados
        for campo, valor in campos_a_modificar.items():
            # Usar un valor predeterminado si el campo no está presente en el YAML original
            if isinstance(nuevo_data[campo],dict):
                nuevo_data[campo].update(valor)
            else:
                nuevo_data[campo] = valor
            print(archivo,campo,valor)

        name = generar_nombre_archivo(nuevo_data)

        # Guardar el archivo modificado
        with open(name, 'w') as filew:
            yaml.dump(dict(nuevo_data), filew, default_flow_style=False)

if __name__ == "__main__":
    # Lista de archivos YAML a modificar
    parser = argparse.ArgumentParser(description='Modificar archivos YAML.')
    parser.add_argument('archivos', nargs='+', help='Archivos YAML a modificar')
    args = parser.parse_args()

    # Campos y valores a modificar
    campos_modificar = {
        "enable":True,
        "training": {
            "timeout":770,
            "max_epochs": 100,
            "patience": 15,
            "iterations": 200,
        }
    }

    # Modificar archivos YAML
    modificar_yaml(args.archivos, campos_modificar)