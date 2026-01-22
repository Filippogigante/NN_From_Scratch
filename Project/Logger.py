import sys
import os

class logger(object):
    def __init__(self):
        self.cartella_attuale = r"C:\Users\nicol\Desktop\Universita\ML\repo\data_weights"
        self.percorso_log = r"C:\Users\nicol\Desktop\Universita\ML\repo\data_weights\grid_search.txt" # Unisce il percorso cartella attuale con il file txt
        self.terminal = sys.stdout
        self.log = open(self.percorso_log, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()
