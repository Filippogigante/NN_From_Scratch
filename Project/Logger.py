import sys
import os

class logger(object):
    def __init__(self):
        '''
        L'output standard di Pyzo viene
        salvato come self.terminal;
        apriamo in modalità "w" (overwrite)
        così cancella il file precedente
        '''
        self.cartella_attuale = r"/Users/filippo/Desktop/data_weights"
        self.percorso_log = r"/Users/filippo/Desktop/NiralNeuorcFromScretch-jaeger/data_weights/grid_search.txt" # Unisce il percorso cartella attuale con il file txt
        self.terminal = sys.stdout
        self.log = open(self.percorso_log, "w", encoding="utf-8")

    def write(self, message):
        '''
        terminal.write scrive nella shell;
        log.write scrive nel file di testo;
        log.flush forza la scrittura immediata su disco
        '''
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        '''
        Si assicura che non
        ci siano dati non scritti
        '''
        self.terminal.flush()
        self.log.flush()
