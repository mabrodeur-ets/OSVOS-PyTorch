from util.path_abstract import PathAbstract

class Path(PathAbstract):
    @staticmethod
    def db_root_dir():
        return 'code/osvos-pytorch/data/DAVIS'      #CCDB
        #return '.\\osvos-pytorch\\data\\DAVIS'     #WIN10

    @staticmethod
    def save_root_dir():
        return 'code/osvos-pytorch/models'          #CCDB
        #return '.\\osvos-pytorch\\models'          #WIN10

    @staticmethod
    def models_dir():
        return "code/osvos-pytorch/models"          #CCDB
        #return '.\\osvos-pytorch\\models'          #WIN10
