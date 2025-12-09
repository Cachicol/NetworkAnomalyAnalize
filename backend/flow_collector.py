import os
import time
import pandas as pd
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler



# ====== CONFIG ======
CIC_DAILY_DIR = r"C:\Users\pichau\CICFlowMeter-master\data\daily"
OUTPUT_CSV = "data/flows.csv"

# Guarda a quantidade de linhas já processadas
last_line_count = 0

def get_today_filename():
    today = datetime.now().strftime("%Y-%m-%d")
    return f"{today}_Flow.csv"


def process_incremental(path):
    global last_line_count

    try:
        df = pd.read_csv(path)
        total_lines = len(df)

        # Se o arquivo diminuiu (resetou), começamos do zero
        if total_lines < last_line_count:
            print("[RESET] Arquivo reiniciado pelo CICFlowMeter. Lendo desde o início.")
            last_line_count = 0

        # Quantas linhas novas apareceram?
        new_lines = total_lines - last_line_count

        if new_lines <= 0:
            return  # nada a fazer

        df_new = df.tail(new_lines).copy()
        df_new["ts_collected"] = datetime.now().isoformat()

        # Append no CSV de saída
        if os.path.exists(OUTPUT_CSV):
            df_new.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
        else:
            df_new.to_csv(OUTPUT_CSV, index=False)

        print(f"[OK] {new_lines} novas linhas adicionadas.")

        # Atualiza o contador
        last_line_count = total_lines

    except Exception as e:
        print(f"[ERRO] Falha ao processar incremento: {e}")


class DailyFlowHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory:
            filename = os.path.basename(event.src_path)
            if filename == get_today_filename():
                process_incremental(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            filename = os.path.basename(event.src_path)
            if filename == get_today_filename():
                process_incremental(event.src_path)




def main():
    global last_line_count

    today_file = get_today_filename()
    today_path = os.path.join(CIC_DAILY_DIR, today_file)

    print("=== Flow Collector V2 (CICFlowMeter Incremental) ===")
    print(f"Monitorando: {CIC_DAILY_DIR}")
    print(f"Arquivo esperado: {today_file}")
    print("Pressione CTRL+C para encerrar.\n")

    # Se o arquivo já existir ao iniciar, contamos as linhas mas não processamos todas
    if os.path.exists(today_path):
        df_existing = pd.read_csv(today_path)
        last_line_count = len(df_existing)
        print(f"[BOOT] Arquivo já tem {last_line_count} linhas. Aguardando novas...")

    event_handler = DailyFlowHandler()
    observer = Observer()
    observer.schedule(event_handler, CIC_DAILY_DIR, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()


if __name__ == "__main__":
    main()
