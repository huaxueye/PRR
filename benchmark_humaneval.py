import time
import subprocess
import os
import re
import datetime
import numpy as np
import argparse
import sys

# ================= Configuration =================
MEMORY_THRESHOLD_MB = 23000  # GPU considered free if free memory > 23GB
POLL_INTERVAL_SECONDS = 10
MAX_CONCURRENT_PER_GPU = 1  # Only 1 job per GPU
DEFAULT_GPUS = [0]  # Default to GPU 0 if not specified

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXEC = sys.executable
SCRIPT_PATH = os.path.join(BASE_DIR, "prr_evaluate.py")
MODEL_PATH = "GSAI-ML/LLaDA-1.5"

# Default Head Path (can be overridden)
DEFAULT_HEAD_PATH = os.path.join(BASE_DIR, "head_checkpoint.pt")
DEFAULT_LOG_SUFFIX = "open_source"

def main():
    parser = argparse.ArgumentParser(description="Dynamic Scheduler for HumanEval Evaluation")
    parser.add_argument("--head_path", type=str, default=DEFAULT_HEAD_PATH, help="Path to the trained head checkpoint")
    parser.add_argument("--log_suffix", type=str, default=DEFAULT_LOG_SUFFIX, help="Suffix for log and result directories")
    parser.add_argument("--gpus", type=str, default=None, help="Comma separated list of GPUs to use (e.g. '0,1')")
    parser.add_argument("--alphas", type=str, default="1.0", help="Comma separated list of alphas")
    parser.add_argument("--thresholds", type=str, default=None, help="Comma separated list of thresholds")
    parser.add_argument("--steps", type=int, default=512, help="Diffusion steps")
    parser.add_argument("--block_length", type=int, default=32, help="Block length")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples (optional)")
    args = parser.parse_args()

    head_path = args.head_path
    log_suffix = args.log_suffix
    
    if args.gpus:
        gpus = [int(x) for x in args.gpus.split(",")]
    else:
        gpus = DEFAULT_GPUS

    print(f"[*] Configuration:")
    print(f"    Head Path: {head_path}")
    print(f"    Log Suffix: {log_suffix}")
    print(f"    GPUs: {gpus}")

    LOG_DIR = os.path.join(BASE_DIR, f"logs_humaneval_{log_suffix}")
    RESULTS_DIR = os.path.join(BASE_DIR, f"results_humaneval_{log_suffix}")

    # Base Args
    # Using weighted strategy
    COMMON_ARGS_BASE = f"model_path={MODEL_PATH},head_path={head_path},gen_length=512,steps={args.steps},block_length={args.block_length},strategy=weighted,force_instruct=False,show_speed=True"
    
    # Ensure dirs exist
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # ================= Task Definitions =================
    tasks = []
    
    def add_task(name, model_args_suffix):
        output_dir = os.path.join(RESULTS_DIR, name)
        final_args = f"{COMMON_ARGS_BASE},save_dir={output_dir},{model_args_suffix}"
        
        # Check if results already exist
        if os.path.exists(output_dir):
            # Check recursively for any .json file
            has_json = False
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.endswith(".json"):
                        has_json = True
                        break
                if has_json:
                    break
            
            if has_json:
                print(f"Skipping {name}: Results already found in {output_dir}")
                return
    
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "results.json")
        log_path = os.path.join(LOG_DIR, f"{name}.log")
        
        cmd = [
            PYTHON_EXEC, "-u", SCRIPT_PATH,
            "--model", "llada_head",
            "--tasks", "humaneval",
            "--num_fewshot", "0",
            "--confirm_run_unsafe_code",
            "--model_args", final_args,
            "--output_path", output_path
        ]

        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        
        tasks.append({
            "name": name,
            "cmd": cmd,
            "log_path": log_path
        })
    
    # Define the grid
    alphas = [float(x) for x in args.alphas.split(",")]
    if args.thresholds:
        thresholds = [float(x) for x in args.thresholds.split(",")]
    else:
        thresholds = np.arange(0.70, 0.95 + 0.001, 0.05)
    
    for alpha in alphas:
        for th in thresholds:
            th_val = round(th, 2)
            task_name = f"weighted_alpha{alpha}_th{th_val}"
            task_args = f"temp_alpha={alpha},temp_threshold={th_val}"
            add_task(task_name, task_args)
    
    print(f"Generated {len(tasks)} tasks.")
    
    # ================= Scheduler Logic =================
    running_procs = {} # gpu_id -> {'proc': subprocess.Popen, 'task': task_dict, 'start_time': datetime}
    event_log = []
    
    def log_event(msg):
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        event_log.append(f"[{timestamp}] {msg.strip()}")
        if len(event_log) > 15:
            event_log.pop(0)
    
    def get_gpu_memory_map():
        """Returns a dict {gpu_id: free_memory_mb}"""
        try:
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,nounits,noheader'],
                encoding='utf-8'
            )
            gpu_memory = {}
            for line in result.strip().split('\n'):
                if not line: continue
                idx, free_mem = line.split(',')
                gpu_memory[int(idx)] = int(free_mem.strip())
            return gpu_memory
        except Exception as e:
            print(f"Error getting GPU memory: {e}")
            return {gpu: 99999 for gpu in gpus} # Fallback
    
    def print_status(done_count, total_count):
        # os.system('clear')
        print(f"=== Dynamic Scheduler (Weighted Strategy) [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ===")
        print(f"Progress: {done_count}/{total_count} tasks completed.")
        print("-" * 60)
        print(f"{'GPU':<5} | {'Status':<10} | {'Task':<40} | {'Duration'}")
        print("-" * 60)
        
        for gpu in gpus:
            if gpu in running_procs:
                info = running_procs[gpu]
                duration = str(datetime.datetime.now() - info['start_time']).split('.')[0]
                print(f"{gpu:<5} | {'BUSY':<10} | {info['task']['name']:<40} | {duration}")
            else:
                print(f"{gpu:<5} | {'IDLE':<10} | {'-':<40} | -")
        print("-" * 60)
        print(f"Queue: {len(tasks)} tasks remaining.")
        print("\n=== Recent Events ===")
        for event in event_log:
            print(event)
    
    print(f"Starting scheduler with {len(tasks)} tasks on GPUs {gpus}...")
    completed_tasks = 0
    total_tasks = len(tasks)
    
    while True:
        # 1. Check running processes
        finished_gpus = []
        for gpu, info in running_procs.items():
            if info['proc'].poll() is not None: # Process finished
                finished_gpus.append(gpu)
                completed_tasks += 1
                if info['proc'].returncode != 0:
                    log_event(f"Task {info['task']['name']} failed on GPU {gpu} with code {info['proc'].returncode}")
                else:
                    log_event(f"Task {info['task']['name']} completed on GPU {gpu}")
        
        for gpu in finished_gpus:
            try:
                running_procs[gpu]['log_file'].close()
            except:
                pass
            del running_procs[gpu]
            
        # 2. Check if all done
        if not tasks and not running_procs:
            log_event("All tasks completed!")
            print_status(completed_tasks, total_tasks)
            break
            
        # 3. Assign new tasks
        if tasks:
            try:
                gpu_memory = get_gpu_memory_map()
                
                # Check each available GPU
                for gpu in gpus:
                    if gpu not in running_procs:
                        # Check if GPU has enough memory
                        if gpu_memory.get(gpu, 0) > MEMORY_THRESHOLD_MB:
                            # Assign task
                            task = tasks.pop(0)
                            
                            # Prepare env
                            env = os.environ.copy()
                            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                            
                            # Open log file
                            os.makedirs(os.path.dirname(task['log_path']), exist_ok=True)
                            log_file = open(task['log_path'], "w")
                            
                            # Start process
                            print(f"Starting task {task['name']} on GPU {gpu}")
                            proc = subprocess.Popen(
                                task['cmd'],
                                stdout=log_file,
                                stderr=subprocess.STDOUT,
                                env=env
                            )
                            
                            running_procs[gpu] = {
                                'proc': proc,
                                'task': task,
                                'start_time': datetime.datetime.now(),
                                'log_file': log_file
                            }
                            
                            log_event(f"Started {task['name']} on GPU {gpu}")
                            
                            if not tasks:
                                break
            except Exception as e:
                print(f"Scheduler error: {e}")
                
        # 4. Print status
        print_status(completed_tasks, total_tasks)
        
        # 5. Sleep
        time.sleep(POLL_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
