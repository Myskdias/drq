from dataclasses import dataclass
import random
import numpy as np
from tqdm import tqdm
from multiprocessing import get_context

from corewar import MARS, Core, redcode

@dataclass
class SimulationArgs:
    rounds: int = 24 # Rounds to play
    size: int = 8000 # The core size
    cycles: int = 80000 # Cycles until tie
    processes: int = 8000 # Max processes
    length: int = 100 # Max warrior length
    distance: int = 100 # Minimum warrior distance

def simargs_to_environment(args):
    return dict(ROUNDS=args.rounds, CORESIZE=args.size, CYCLES=args.cycles,
                MAXPROCESSES=args.processes, MAXLENGTH=args.length, MINDISTANCE=args.distance)

class MyMARS(MARS):
    def __init__(self, core=None, warriors=None, minimum_separation=100, randomize=True, max_processes=None):
        self.core = core if core else Core()
        self.minimum_separation = minimum_separation
        self.max_processes = max_processes if max_processes else len(self.core)
        self.warriors = warriors if warriors else []
        self.warrior_cov = {w: np.zeros(len(self.core), dtype=bool) for w in self.warriors}
        # self.warrior_tsp = {w: 0 for w in self.warriors}  # total spawned processes

        if self.warriors:
            self.load_warriors(randomize)

    def core_event(self, warrior, address, event_type):
        i = self.core[address]
        assert isinstance(i.a_number, int), f"Expected int but got {type(i.a_number)}"
        assert isinstance(i.b_number, int), f"Expected int but got {type(i.b_number)}"
        i.a_number = min(max(i.a_number, -999999999), 999999999)
        i.b_number = min(max(i.b_number, -999999999), 999999999)
        assert i.a_number < 1000000000 and i.a_number > -1000000000, f"a_number out of bounds"
        assert i.b_number < 1000000000 and i.b_number > -1000000000, f"b_number out of bounds"
    
    def enqueue(self, warrior, address):
        """Enqueue another process into the warrior's task queue. Only if it's
           not already full.
        """
        if len(warrior.task_queue) < self.max_processes:
            warrior.task_queue.append(self.core.trim(address))
            self.warrior_cov[warrior][self.core.trim(address)] = True
            # self.warrior_tsp[warrior] += 1

def run_single_round(simargs, warriors, seed, pbar=False):
    random.seed(seed)
    simulation = MyMARS(warriors=warriors, minimum_separation=simargs.distance, max_processes=simargs.processes, randomize=True)
    score = np.zeros(len(warriors), dtype=float)
    alive_score = np.zeros(len(warriors), dtype=float)

    prev_nprocs = np.array([len(w.task_queue) for w in simulation.warriors], dtype=int)
    total_spawned_procs = np.zeros(len(simulation.warriors), dtype=int)

    # memory_coverage = [set(w.task_queue) for w in simulation.warriors]
    for t in tqdm(range(simargs.cycles), disable=not pbar):
        simulation.step()

        nprocs = np.array([len(w.task_queue) for w in simulation.warriors], dtype=int)

        alive_flags = (nprocs>0).astype(int)
        n_alive = alive_flags.sum()
        if n_alive==0:
            break
        score += (alive_flags * (1./n_alive)) / simargs.cycles
        alive_score += alive_flags / simargs.cycles

        total_spawned_procs = total_spawned_procs + np.maximum(0, nprocs - prev_nprocs)
        prev_nprocs = nprocs

        # memory_coverage = [mc.union(set(w.task_queue)) for mc, w in zip(memory_coverage, simulation.warriors)]
    # memory_coverage = np.array([len(mc) for mc in memory_coverage], dtype=int)
    memory_coverage = np.array([cov.sum() for cov in simulation.warrior_cov.values()], dtype=int)
    score = score * len(warriors)
    outputs = dict(score=score, alive_score=alive_score, total_spawned_procs=total_spawned_procs, memory_coverage=memory_coverage)
    return outputs

def _run_battle_task(task):
    simargs, warriors, seed = task
    return run_single_round(simargs, warriors, seed)

def run_battle_batch(simargs, battles, seeds, n_processes=1, timeout=900):
    seeds = list(seeds)
    tasks = [(simargs, warriors, seed) for warriors in battles for seed in seeds]
    # DRQ may initialize CUDA for the direct Hugging Face backend before
    # launching simulator workers. `spawn` avoids inheriting that CUDA runtime.
    with get_context("spawn").Pool(processes=n_processes) as pool:
        result = pool.map_async(_run_battle_task, tasks)
        outputs = result.get(timeout=timeout)

    grouped = []
    for i in range(len(battles)):
        battle_outputs = outputs[i * len(seeds):(i + 1) * len(seeds)]
        grouped.append({
            key: np.stack([output[key] for output in battle_outputs], axis=-1)
            for key in battle_outputs[0].keys()
        })
    return grouped

def run_multiple_rounds(simargs, warriors, n_processes=1, timeout=900, seeds=None):
    try:
        seeds = list(range(simargs.rounds)) if seeds is None else list(seeds)
        return run_battle_batch(
            simargs, [warriors], seeds, n_processes=n_processes, timeout=timeout
        )[0]
    except Exception as e:
        print(e)
        return None

def parse_warrior_from_file(simargs, file):
    environment = simargs_to_environment(simargs)
    with open(file, encoding="latin1") as f:
        warrior_str = f.read()
    warrior = redcode.parse(warrior_str.split("\n"), environment)
    return warrior_str, warrior

