import re
import psutil
import time 
from datetime import datetime
import re 
import pandas as pd
import os
import os
import re
def get_cfs_features(pid) : 
    features = {
         "vruntime": None,
        "actual_cpu_runtime": None,
        "migration_count": None,
        "scheduler_switches": None,
        "voluntary_switches": None,
        "involuntary_switches": None,
        "scheduler_weight": None,
        "runnable_avg": None 
    }
    try : 
        with open(f"/proc/{pid}/sched", "r") as f:
            data = f.read()
       
        patterns = {
            "vruntime": r"se\.vruntime\s*:\s*([\d.]+)",
            "actual_cpu_runtime": r"se\.sum_exec_runtime\s*:\s*([\d.]+)",
            "migration_count": r"se\.nr_migrations\s*:\s*(\d+)",
            "scheduler_switches": r"nr_switches\s*:\s*(\d+)",
            "voluntary_switches": r"nr_voluntary_switches\s*:\s*(\d+)",
            "involuntary_switches": r"nr_involuntary_switches\s*:\s*(\d+)",
            "scheduler_weight": r"se\.load\.weight\s*:\s*(\d+)",
            "runnable_avg": r"se\.avg\.runnable_avg\s*:\s*(\d+)"
        }
        for name , pattern  in patterns.items() : 
            match = re.search(pattern , data)
            if match: 
                features[name] = float( match.group(1))
    except  (FileNotFoundError , PermissionError)  : 
         pass
    return features 
               
previous_wait_ns = {}
def get_task_wait_delta_ms(pid, previous_wait_ns):
    try:
        with open(f"/proc/{pid}/schedstat", "r") as f:
            values = f.read().split()

        current_ns = int(values[1])
        previous_ns = previous_wait_ns.get(pid)
        previous_wait_ns[pid] = current_ns

        if previous_ns is None or current_ns < previous_ns:
            return None

        return (current_ns - previous_ns) / 1_000_000

    except (OSError, ValueError, IndexError):
        return None 


def get_cpu_queue_lengths():
    queues = {}
    current_cpu = None

    try:
        with open("/sys/kernel/debug/sched/debug", "r") as f:
            for line in f:
                cpu_match = re.match(r"^\s*cpu#(\d+)", line)

                if cpu_match:
                    current_cpu = int(cpu_match.group(1))
                    continue

                queue_match = re.match(
                    r"^\s+\.nr_running\s*:\s*(\d+)",
                    line
                )

                if (
                    queue_match
                    and current_cpu is not None
                    and current_cpu not in queues
                ):
                    queues[current_cpu] = int(queue_match.group(1))

    except OSError as e:
        print(f"Queue collection failed: {e}")

    return {
        f"queue_length_cpu_{i}": queues.get(i)
        for i in range(16)
    }
    
        
records = []
duration = 3600
interval = 1 
start_time = time.time()
while time.time() - start_time < duration :
    timestamp = datetime.now()
    
    cpu_usage = psutil.cpu_percent(
        interval= interval , 
        percpu=  True
    )
    memory = psutil.virtual_memory()
    queue_features = get_cpu_queue_lengths()
   
    overall_cpu = sum(cpu_usage) / len(cpu_usage)
    
    processes = list(
        psutil.process_iter(
            [
                'pid',
                'name',
                'cpu_percent',
                'memory_percent',
                'nice',
                'cpu_num',
                'num_ctx_switches',
                'create_time',
                'status'
            ]
        )
    )
    process_count = len(processes)
    running_processes = 0 
    for p in processes : 
        try : 
            if p.status()  == psutil.STATUS_RUNNING : 
                running_processes += 1 
        except (psutil.NoSuchProcess , psutil.AccessDenied) :
            pass
     ## system row 
  
    ## core   
    for  p  in  processes : 
         try : 
             info  = p.info
             pid = info['pid']
             cfs = get_cfs_features(pid)
             task_wait_time = get_task_wait_delta_ms(pid , previous_wait_ns)
             
             row = {
                  "timestamp": timestamp,

                "overall_cpu_utilization": overall_cpu,

                "memory_percent": memory.percent,

                "memory_used_mb":
                    memory.used / (1024 * 1024),

                "process_count":
                    process_count,

                "running_processes":
                    running_processes
             }
             row.update(queue_features)
             for i in range(16) : 
                 row[f"cpu_{i}_utilization"]  = (
                  cpu_usage[i] if   len(cpu_usage) else None
               )
             row["task_id"] = pid
             row["task_wait_time_delta_ms"] = task_wait_time

             row["task_name"] = info["name"]

             row["task_cpu_usage"] = info["cpu_percent"]

             row["task_memory_usage"] = info["memory_percent"]
             row["task_runtime"] = time.time() - info["create_time"]
             
             row["task_state"] = info["status"]
             row["task_priority"] = info["nice"]

             row["current_core"] = info["cpu_num"]  
             ctx =  info["num_ctx_switches"]
             if ctx : 
                 row["context_switches"] = (
                     ctx.voluntary + ctx.involuntary
                 )
             else : 
                 row["context_switches"] = None
             row["vruntime"] = cfs["vruntime"]

             row["actual_cpu_runtime"] = (
                cfs["actual_cpu_runtime"]
            )

             row["migration_count"] = (
                cfs["migration_count"]
            )

             row["scheduler_switches"] = (
                cfs["scheduler_switches"]
            )

             row["voluntary_switches"] = (
                cfs["voluntary_switches"]
            )

             row["involuntary_switches"] = (
                cfs["involuntary_switches"]
            )

             row["scheduler_weight"] = (
                cfs["scheduler_weight"]
            )

             row["runnable_avg"] = (
                cfs["runnable_avg"]
            )
             records.append(row)
         except (
            psutil.NoSuchProcess , 
            psutil.AccessDenied , 
            psutil.ZombieProcess
        )     :
            pass
          



df = pd.DataFrame(records)
print(df.head())
print(df.shape)
df.to_csv(
    "ai_multicore_scheduler_dataset_fnf.csv",
    index=False
)


