import training
import optuna
import utils as ut
import dataflow_own as own
import multiprocessing
import sys, os

def run(options):

    # IMPORT RAW DATA
    rawdata = ut.read_data(options['paths']['root']+options['paths']['raw_data'])
    # options["data"]["scaler"] = Normalizer

    def objective(trial):
        model = training.MLP(trial,Dataset, options["experiment_id"])
        metric = model.optimize_with_optuna(trial, options["training"])
        print("Accuracy: ", metric)
        return metric

    # CHOOSE EXPERIMENT
    if options["experiment"]=="own":
        Dataset = own.Dataflow_own(rawdata,options)
        print(f"INFO - Exp {options['experiment_id']}: Dataset created",end='\n')

        # Create a study and optimize the model
        study = optuna.create_study(direction=options["optimization"]["direction"])
        study.optimize(objective, n_trials=options["optimization"]["iterations"])

        # Get the best hyperparameters
        best_params = study.best_trial.params
        print(f"INFO - Exp {options['experiment_id']}: Best Hyperparameters:", best_params,end='\n')

        # Create the MLP model with the best hyperparameters
        best_model = training.MLP(study.best_trial,Dataset, options["experiment_id"])

        print(f"INFO - Exp {options['experiment_id']}: Process logs metrics",end='\n')
        ut.save_metrics(options["experiment_id"])

        print(f"INFO - Exp {options['experiment_id']}: Process logs predictions",end='\n')
        ut.save_outputs(options["experiment_id"],Dataset.df)

        print(f"INFO - Exp {options['experiment_id']}: Removing unnecessary log files",end='\n')
        ut.delete_event_logs(options["experiment_id"])

        print(f"INFO - Exp {options['experiment_id']}: Done!\n\n",end='\n')

if __name__=="__main__":
    if sys.platform=='win32':
        root = "F:\\TFG\\"
        path_pending = "F:\\TFG\\code\\experiments\\_pending\\"
        path_done = "F:\\TFG\\code\\experiments\\_done\\"
        multiprocessing.set_start_method('spawn', force=True)
    else: 
        root = "/home/gti/" 
        path_pending = "/home/gti/experiments/_pending/"
        path_done = "/home/gti/experiments/_done/"

    # Spawn multiple processes for training or evaluation
    pending_configs = os.listdir(path_pending)
    for config in pending_configs:
        options = ut.load_yaml(f"{path_pending}{config}")
        options["paths"]["root"] = root
        process = multiprocessing.Process(target=run,args=(options,))
        process.start()
        process.join()
        if process.exitcode==0: os.replace(f"{path_pending}{config}", f"{path_done}{config}")
