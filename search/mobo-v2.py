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
NUM_SAMPLES=64

#######################################################################
# (1) StepFunctionsProblem: (X)->( -time, -cost )
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
    # 1) Step Functions 입력으로 alias 설정
    payload = {}
    for i, mem in enumerate(mem_list, start=1):
        payload[f"F{i}Alias"] = f"{int(mem)}MB"

    # 2) Step Functions 실행
    start_time = time.time()
    response = sf_client.start_execution(
        stateMachineArn="arn:aws:states:us-east-1:891376968462:stateMachine:ml-pipeline",
        input=json.dumps(payload)
    )
    exec_arn = response["executionArn"]

    # 3) 실행 완료 대기
    while True:
        desc = sf_client.describe_execution(executionArn=exec_arn)
        if desc["status"] in ["SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"]:
            break
        time.sleep(1)

    end_time = time.time()
    wall_clock_sec = end_time - start_time
    
    # 기본값
    total_time = wall_clock_sec
    total_cost = 9999.0

    if desc["status"] == "SUCCEEDED":
        try:
            output_data = json.loads(desc.get("output", "{}"))
            # 최종 상태 output에서 functionTimes, ex: {"F1":1.2, "F2":..., "F3":..., "F4":...}
            func_times = output_data.get("functionTimes", None)
            if func_times is not None:
                # 4) 각 함수 실행 시간을 합산 (순차 구조 가정)
                #    total_time = sum(func_times.values())
                #    or, 만약 'exec_time_s' 4개를 그냥 더하는 경우
                sum_time = 0.0
                for key in sorted(func_times.keys()):
                    sum_time += float(func_times[key])
                total_time = sum_time

                # 5) 비용 계산: sum( time_i * (mem_list[i]/1024) * rate )
                rate = 0.00001667
                cost_sum = 0.0
                # mem_list[i]에 대응 -> F{i}
                for i, key in enumerate(sorted(func_times.keys()), start=0):
                    f_time = float(func_times[key])  # sec
                    f_mem = float(mem_list[i])
                    cost_sum += f_time * (f_mem/1024.0) * rate
                total_cost = cost_sum
            else:
                total_time = wall_clock_sec
                total_cost = 9999.0

        except Exception:
            total_time = 999999.0
            total_cost = 9999.0
    else:
        # 실패/timeout
        total_time = 999999.0
        total_cost = 9999.0

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
# (2) 초기 데이터 / 모델 초기화
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

    gp_time = SingleTaskGP(Xn, y_time)
    gp_cost = SingleTaskGP(Xn, y_cost)

    model = ModelListGP(gp_time, gp_cost)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    
    problem.ref_point = torch.tensor([-600.0, -10.0], dtype=torch.double)
    return mll, model

#######################################################################
# (3) qNEHVI optimize
#######################################################################
def optimize_acqf_and_get_observation(model, train_x, problem, batch_size=2, restarts=5, raw_samples=64, sampler = None):
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
# (4) BO 루프
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
            # (optional)
            print(
            f"\nBatch {iteration:>2}: Hypervolume (qLNEHVI) = "
            f"({hvs[-1]:>4.2f}), time = {t1-t0:>4.2f}.",
            end="",
        )
        else:
            print(".", end="")

    return train_x, train_obj

#######################################################################
# (5) main
#######################################################################
if __name__=="__main__":
    final_x, final_y = bo_loop(
        dim=4,
        n_init=3,       # 초기 탐색 점
        n_batch=20,     # 총 이터레이션
        batch_size=3,   # 배치 크기
        verbose=True
    )
    print("=== Done ===")
    print(f"Collected {final_x.shape[0]} points")
    print("final_y sample:\n", final_y[:5])
