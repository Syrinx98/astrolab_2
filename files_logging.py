#!/usr/bin/env python3
import os
import subprocess
import sys

def main():
    # nome del tuo script da escludere
    script_name = 'files_logging.py'

    # directory in cui si trova questo script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # crea cartella per i log
    logs_dir = os.path.join(script_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # percorso del file di log unico
    combined_log = os.path.join(logs_dir, 'combined.log')

    # 1) creazione del file con prova di scrittura
    with open(combined_log, 'w', encoding='utf-8') as log_file:
        log_file.write("=== Avvio del log ===\n")
        log_file.write("Prova di scrittura riuscita ✔\n")

    # 2) esecuzione di tutti i .py (escluso logging.py) uno a uno,
    #    scrivendo man mano dentro combined.log
    py_files = [
        f for f in os.listdir(script_dir)
        if f.endswith('.py') and f != script_name
    ]
    py_files.sort()  # ordine lessicografico

    with open(combined_log, 'a', encoding='utf-8') as log_file:
        for py in py_files:
            log_file.write(f"\n===== Esecuzione di: {py} =====\n")
            input_path = os.path.join(script_dir, py)

            # esegui lo script
            result = subprocess.run(
                [sys.executable, input_path],
                capture_output=True,
                text=True
            )

            # scrivi stdout
            log_file.write("--- stdout ---\n")
            log_file.write(result.stdout or "(nessun output)\n")
            # scrivi stderr, se presente
            if result.stderr:
                log_file.write("\n--- stderr ---\n")
                log_file.write(result.stderr)

        log_file.write("\n===== FINE DI TUTTE LE ESECUZIONI =====\n")

    print(f"Tutti i log sono stati salvati in: {combined_log}")

if __name__ == "__main__":
    main()
