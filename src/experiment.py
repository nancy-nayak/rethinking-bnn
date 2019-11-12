# Functions related to sacred experiment

import sacred

class Experiment(sacred.Experiment):
    def __init__(self, name, dataset, model, hparams, observer):
        """
            Create a scared experiment with the given settings
            Options:
                name    : Name of Experiment (string)
                dataset : Dataset name (string)
                model   : Model name (string)
                hparams : Dictionary of hyper parameters
                observer: Experiment Observer. 
                    See https://sacred.readthedocs.io/en/stable/observers.html
                    Ex: mongo_db=127.0.0.1:27017
                        file_storage=./logs
        """

        # Initilize Sacred experiment
        super().__init__(name=name)
        
        # Build sacred configuration for this experiment
        ex_cfg = {}
        for (param, value) in hparams.items():
            if not callable(value): # Filter out all non callable parameters
                if type(value) is dict:
                    # If the hyper-parameter is specified as a dictionary,
                    # then unpack it with proper names
                    for(_param, _value) in value.items():
                        ex_cfg[f"{param}:{_param}"] = _value
                else:
                    ex_cfg[param] = value
        ex_cfg["dataset"] = dataset
        ex_cfg["model"] = model
        
        # To disable logging of output to screen
        self.captured_out_filter = lambda captured_output: "Output capturing turned off." 
        
        # @TODO: Disable source file logging or point to called script
        # self.add_source_file("")
        
        # Add the configuration
        self.add_config(ex_cfg)

        # Save the observer
        self.observer = observer

    def execute(self):
        # build argv for sacred -- hacky way!
        _argv = f"{self.default_command} --{self.observer}"
        # _argv = f"{self.default_command}"
        self.run_commandline(argv=_argv)
