from util.path_abstract import PathAbstract

class Path(PathAbstract):
    @staticmethod
    def db_root_dir():
        return 'code/osvos-pytorch/data/DAVIS' # return '.\\osvos-pytorch\\data\\DAVIS'

    @staticmethod
    def save_root_dir():
        return 'code/osvos-pytorch/models' # return '.\\osvos-pytorch\\models'

    @staticmethod
    def models_dir():
        return "code/osvos-pytorch/models" # return '.\\osvos-pytorch\\models'
