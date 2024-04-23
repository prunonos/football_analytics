import pandas as pd
from dataset_owntop1 import Dataset_OwnTop1
from lgbm_top1 import Lgbm_top1
from own_top1 import Own_top1
from tabnet_owntop1 import Tabnet_Owntop1
from tabnet_piRatings import Tabnet_piRatings
from tabnet_own import Tabnet_Own
from own_original import Own_Original
from dataset_own import Dataset_Own
from dataset_top1 import Dataset_top1
from dataset_piRatings import Dataset_piRatings
import utils as ut
import multiprocessing
import sys, os, time

def run(options,dtypes):
    # IMPORT RAW DATA
    rawdata = ut.read_data(options['paths']['raw_data'],dtypes=dtypes)

    pd.set_option('mode.chained_assignment', None)

    # CHOOSE EXPERIMENT
    if options["experiment"]=="own":
        pass
    elif options["experiment"]=='lgbm_top1':
        print(f"INFO - Running LightGBM top1: {options['experiment_id']} - {options['description']}")
        dataset = Dataset_top1(rawdata,options)
        Lgbm_top1(dataset,options)
    elif options["experiment"]=='tabnet_own':
        print(f"INFO - Running Tabnet Own: {options['experiment_id']} - {options['description']}")
        dataset = Dataset_Own(rawdata,options)
        Tabnet_Own(dataset,options)
    elif options["experiment"]=='tabnet_paper':
        print(f"INFO - Running Tabnet Paper: {options['experiment_id']} - {options['description']}")
        dataset = Dataset_piRatings(rawdata,options)
        Tabnet_piRatings(dataset,options)
    elif options["experiment"]=='own_original':
        print(f"INFO - Running Own experiment: {options['experiment_id']} - {options['description']}")
        dataset = Dataset_Own(rawdata,options)
        Own_Original(dataset,options)
    elif options["experiment"]=='own_top1':
        print(f"INFO - Running Own model with top1 data with DL nets: {options['experiment_id']} - {options['description']}")
        dataset = Dataset_OwnTop1(rawdata,options)
        Own_top1(dataset,options)
    elif options["experiment"]=='tabnet_owntop1':
        print(f"INFO - Running Tanet model with top1 data: {options['experiment_id']} - {options['description']}")
        dataset = Dataset_OwnTop1(rawdata,options)
        Tabnet_Owntop1(dataset,options)

    print(f"INFO - Finished Succesfully: {options['experiment_id']} - {options['description']}")

def process_yaml(queue_path, done_path, queue_name):
    cont = 0
    while True:
        cont += 1
        # Obtener el siguiente YAML de la cola
        pending_configs = sorted(os.listdir(queue_path))
        pending_configs_with_priority = []

        for config in pending_configs:
            if config.endswith(".yaml"):
                options = ut.load_yaml(f"{queue_path}{config}")
                # Verificar si el YAML pertenece a la cola actual
                if options.get('queue',"default") == queue_name and options["enable"]:
                    priority = options.get('priority', 2)  # Prioridad predeterminada: 2
                    pending_configs_with_priority.append((priority, config, options))

        # Ordenar por prioridad descendente
        pending_configs_with_priority.sort(reverse=True, key=lambda x: x[0])
        cont = 0 if len(pending_configs_with_priority) else cont

        for _, config, options in pending_configs_with_priority:
            dtypes = ut.load_json(root + options['paths']['dtypes_path'])
            for key in options["paths"].keys():
                options["paths"][key] = root + options["paths"][key]
            options["paths"]["root"] = root

            # Ejecutar el proceso
            process = multiprocessing.Process(target=run, args=(options, dtypes))
            process.start()
            process.join()

            # Mover el YAML a la carpeta de completados si se ejecutó con éxito
            if process.exitcode == 0:
                os.replace(f"{queue_path}{config}", f"{done_path}{config}")

        if cont > 5: break
        time.sleep(60 * 5)

if __name__=="__main__":
    if sys.platform=='win32':
        root = "F:\\TFG\\"
        path_pending = os.getcwd() + "\\_pending\\"
        path_done = os.getcwd() + "\\_done\\"
        multiprocessing.set_start_method('spawn', force=True)
    else: 
        root = "/home/gti/" 
        path_pending = "/home/gti/framework/_pending/"
        path_done = "/home/gti/framework/_done/"

    queue_name = sys.argv[1] if len(sys.argv) > 1 else "default"
    # Spawn multiple processes for training or evaluation
    process_yaml(path_pending,path_done,queue_name)    