import gevent  # isort:skip
from gevent import monkey  # isort:skip
monkey.patch_all()  # isort:skip

import argparse
import json
import logging
import os
import signal
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP, ModelListGP
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.utils.multi_objective.pareto import is_non_dominated

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = torch.device("cpu")
DTYPE = torch.double
tkwargs = {
    "dtype": DTYPE,
    "device": DEVICE
}
# SMOKE_TEST = os.environ.get("SMOKE_TEST")
NOISE_SE = torch.tensor([1.0, 0.05], **tkwargs)
NUM_SAMPLES=128

#######################################################################
# StepFunctionsProblem: (X)->( -time, -cost )
#######################################################################
import boto3

sf_client = boto3.client("stepfunctions", region_name="us-east-1")

import json
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import boto3
import numpy as np

sf_client = boto3.client('stepfunctions', region_name='us-east-1')

def find_function_times(data):
    """
    Recursively search for 'functionTimes' in nested dictionaries.
    """
    if isinstance(data, dict):
        if "functionTimes" in data:
            return data["functionTimes"]
        for key, value in data.items():
            result = find_function_times(value)
            if result:
                return result
    return None

def run_stepfunctions_workflow(mem_list):
    """
    mem_list: ex) [m1, m2, m3, m4] (각 함수의 메모리 MB)
    
    1) Step Functions를 실행하고,
    2) 마지막 상태(Output)에서 "functionTimes" 딕셔너리를 파싱
        => 예: {"F1":1.23, "F2":0.78, "F3":2.14, "F4":0.36}
    3) 각 함수별 (time * (mem/1024) * rate)로 비용 계산
    4) 총 실행 시간 = 각 함수 time 합(순차 구조 가정)
    5) (total_time, total_cost) 반환
    """
    # print("Debug: Starting Step Functions workflow")
    payload = {f"F{i}Alias": f"{int(mem)}MB" for i, mem in enumerate(mem_list, start=1)}

    start_time = time.time()
    response = sf_client.start_execution(
        stateMachineArn="arn:aws:states:us-east-1:891376968462:stateMachine:ml-pipeline",
        input=json.dumps(payload)
    )
    exec_arn = response["executionArn"]
    # print(f"Debug: Execution ARN: {exec_arn}")

    while True:
        desc = sf_client.describe_execution(executionArn=exec_arn)
        if desc["status"] in ["SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"]:
            # print(f"Debug: Execution ended with status {desc['status']}")
            break
        time.sleep(1)

    end_time = time.time()
    wall_clock_sec = end_time - start_time
    total_time = wall_clock_sec
    total_cost = 9999.0

    if desc["status"] == "SUCCEEDED":
        try:
            output_data = json.loads(desc.get("output", "{}"))
            # print(f"Debug: Full Output Data: {json.dumps(output_data, indent=2)}")
            func_times = find_function_times(output_data)
            if func_times:
                sum_time = sum(float(func_times[key]) for key in func_times)
                total_time = sum_time

                rate = 0.00001667
                cost_sum = sum(float(func_times[key]) * (mem_list[i] / 1024.0) * rate
                                for i, key in enumerate(func_times))
                total_cost = cost_sum
                # print("Debug: Successfully calculated time and cost")
            else:
                print("Debug: functionTimes not found in output")
        except Exception as e:
            print(f"Debug: Exception occurred: {e}")
            total_time = 999999.0
            total_cost = 9999.0
    else:
        print("Debug: Inside non-success branch")
        total_time = 999999.0
        total_cost = 9999.0

    # print(f"Debug: Returning total_time={total_time}, total_cost={total_cost}")
    return total_time, total_cost


class StepFunctionsProblem:
    def __init__(self, dim=4):
        self.dim = dim
        self.step = 64.0  # 64MB 간격
        lb = [128.0]*dim  # 최소 메모리
        ub = [3008.0]*dim  # 최대 메모리
        self.bounds = torch.tensor([lb, ub], **tkwargs)

    def __call__(self, X: torch.Tensor):
        """
        X shape: (N, dim)
        return shape (N,2): => [-time, -cost]
        메모리 값을 64MB 간격으로 반올림
        """
        out = []
        # 64MB 간격으로 반올림
        X_rounded = torch.round(X / self.step) * self.step
        # 범위 제한
        X_clipped = torch.clamp(X_rounded, self.bounds[0], self.bounds[1])
        
        arr = X_clipped.cpu().detach().numpy()
        for i in range(arr.shape[0]):
            mems = arr[i,:]
            t_s, c_s = run_stepfunctions_workflow(mems)
            out.append([-t_s, -c_s])
        return torch.tensor(out, **tkwargs)

#######################################################################
# 초기 데이터 / 모델 초기화
#######################################################################
def generate_initial_data(problem, n_init=3):
    """
    n_init 개의 초기 점을 [problem.bounds] 범위에서 sobol
    """
    train_x = draw_sobol_samples(bounds=problem.bounds, n=n_init, q=1).squeeze(1)  # shape(n_init, dim)
    train_obj_true = problem(train_x)  # shape(n_init, 2) => [-time, -cost]
    train_obj = train_obj_true + torch.randn_like(train_obj_true) * 0.1 # add noise
    return train_x, train_obj, train_obj_true

def initialize_model(train_x, train_obj, problem):
    """
    2D output => ModelListGP( time, cost ) => negative 값
    y0= -time, y1= -cost
    """
    # normalize X
    Xn = (train_x - problem.bounds[0]) / (problem.bounds[1]-problem.bounds[0])
    
    y_time = train_obj[...,0:1]  # shape(n,1)
    y_cost = train_obj[...,1:2]  # shape(n,1)

    # manually normalize y_time (삭제할 수도)
    mean_time = y_time.mean()
    std_time = y_time.std()
    y_time = (y_time - mean_time) / (std_time + 1e-9)

    # manually normalize y_cost (삭제할 수도)
    mean_cost = y_cost.mean()
    std_cost = y_cost.std()
    y_cost = (y_cost - mean_cost) / (std_cost + 1e-9)

    gp_time = SingleTaskGP(Xn, y_time)
    gp_cost = SingleTaskGP(Xn, y_cost)

    model = ModelListGP(gp_time, gp_cost)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    
    problem.ref_point = torch.tensor([-600.0, -10.0], dtype=torch.double)
    return mll, model

#######################################################################
# qLNEHVI optimize
#######################################################################
def optimize_acqf_and_get_observation(model, train_x, problem, batch_size=2, restarts=10, raw_samples=128, sampler = None):
    # ref point ( -600, -10 ) etc. => time=600s, cost=$10 worse
    ref_point = torch.tensor([-600.0, -10.0], dtype=torch.double)
    qnehvi = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),
        X_baseline=normalize(train_x, problem.bounds),     # unnormalized -> need to pass normalized version
        sampler=sampler,
        prune_baseline=True,
    )
    # optimize
    dim_ = problem.dim
    bounds_01 = torch.stack([
        torch.zeros(dim_, dtype=torch.double),
        torch.ones(dim_, dtype=torch.double)
    ])
    candidates, _ = optimize_acqf(
        acq_function=qnehvi,
        bounds=bounds_01,
        q=batch_size,
        num_restarts=restarts,
        raw_samples=raw_samples,
        options={"batch_limit":5, "maxiter":200},
        sequential=True,
    )
    
    new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    new_obj_true = problem(new_x)
    new_obj = new_obj_true + torch.randn_like(new_obj_true) * NOISE_SE
    return new_x, new_obj, new_obj_true

#######################################################################
# Visulaization
#######################################################################
def visualize_results(y, hvs):
    """
    Visualize Pareto front and hypervolume progression, then save to file.
    y: Final objectives (Tensor, shape: [N, 2])
    hvs: List of hypervolumes at each batch
    """
    mask = is_non_dominated(y)  # BoTorch 함수
    y_nd = y[mask].cpu().numpy()

    # 예: 두 번째 컬럼(-cost) 기준 정렬
    sorted_idx = y_nd[:, 1].argsort()
    pareto_front = y_nd[sorted_idx]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(-pareto_front[:, 0], -pareto_front[:, 1], 'o-')
    plt.xlabel("Execution Time (s)")
    plt.ylabel("Cost ($)")
    plt.title("Pareto Front: Execution Time vs. Cost")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(hvs) + 1), hvs, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Hypervolume")
    plt.title("Hypervolume Progression")
    plt.grid()

    plt.tight_layout()
    plt.savefig("results.png")
    plt.close()
    
#######################################################################
# pareto front
#######################################################################
    
def get_pareto_inputs(train_x, train_obj_true):
    """
    Get the input values corresponding to Pareto front points.
    """
    pareto_mask = is_non_dominated(train_obj_true)  # Identify Pareto points
    pareto_inputs = train_x[pareto_mask]           # Get inputs for Pareto points
    pareto_outputs = train_obj_true[pareto_mask]   # Get outputs for Pareto points
    return pareto_inputs, pareto_outputs    

#######################################################################
# BO 루프
#######################################################################
def bo_loop(
    dim=4,
    n_init=3,        # 초기 점
    n_batch=20,      # 총 이터레이션
    batch_size=3,    # 배치 크기
    verbose=True
):
    hvs = []
    problem = StepFunctionsProblem(dim=dim)

    # init data
    train_x, train_obj, train_obj_true = generate_initial_data(problem, n_init=n_init)
    mll, model = initialize_model(train_x, train_obj, problem)
    
    # compute hypervolume
    bd = DominatedPartitioning(ref_point=problem.ref_point, Y=train_obj_true)
    volume = bd.compute_hypervolume().item()
    hvs.append(volume)

    # main loop
    for iteration in range(1, n_batch+1):
        t0 = time.monotonic()
        
        # fit the models
        fit_gpytorch_mll(mll)
        
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([NUM_SAMPLES]))
        # qNEHVI
        new_x, new_obj, new_obj_true = optimize_acqf_and_get_observation(
            model=model,
            train_x=(train_x - problem.bounds[0])/(problem.bounds[1]-problem.bounds[0]),
            problem=problem,
            batch_size=batch_size,
            sampler = sampler
        )
        # update training points
        train_x = torch.cat([train_x, new_x])
        train_obj = torch.cat([train_obj, new_obj])
        train_obj_true = torch.cat([train_obj_true, new_obj_true])
        
        # update progress
        bd = DominatedPartitioning(ref_point=problem.ref_point, Y=train_obj_true)
        volume = bd.compute_hypervolume().item()
        hvs.append(volume)

        # re-init
        mll, model = initialize_model(train_x, train_obj, problem)
        t1 = time.monotonic()
        
        if verbose:
            # calculate HV or best cost ...
            print(
            f"\nBatch {iteration:>2}: Hypervolume (qLNEHVI) = "
            f"({hvs[-1]:>4.2f}), time = {t1-t0:>4.2f}.",
            end="",
        )
        else:
            print(".", end="")
            
    # Visualize results
    visualize_results(train_obj_true, hvs)
    
    return train_x, train_obj

#######################################################################
# main
#######################################################################
if __name__=="__main__":
    final_x, final_y = bo_loop(
        dim=4,
        n_init=10,       # 초기 탐색 점
        n_batch=90,     # 총 이터레이션
        batch_size=1,   # 배치 크기
        verbose=True
    )
    print("=== Done ===")
    print(f"Collected {final_x.shape[0]} points")
    print("final_y sample:\n", final_y[:5])

    # Get Pareto inputs and outputs
    pareto_inputs, pareto_outputs = get_pareto_inputs(final_x, final_y)
    print("\nPareto Front Inputs:\n", pareto_inputs)
    print("\nPareto Front Outputs:\n", pareto_outputs)