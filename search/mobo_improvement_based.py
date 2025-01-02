import boto3
import torch
import time
import json
import warnings

import numpy as np
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim.optimize import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)

# Step Functions boto3 client
sf_client = boto3.client('stepfunctions', region_name='us-east-1')

########################################################################
# 1. run_stepfunctions_workflow:
#    - total_time: 로그 정보 기반 / 실패 시 wall-clock fallback
#    - total_cost: 람다 비용 공식 사용 (간단 예시)
########################################################################
def run_stepfunctions_workflow(mem_list):
    """
    mem_list: ex) [m1, m2, m3, m4]  (함수 개수만큼)
    returns: (time_s, cost)

    time_s: 로그에서 파싱 성공 시 그 값; 안 되면 wall-clock sec
    cost:  람다 비용 = sum_i ( (mem_i MB / 1024) * (duration_i sec) * rate )
          여기선 전체 time_s * (avg mem_list / 1024) * 0.0000167
          (실제론 함수별 duration 각각 곱)
    """
    payload = {}
    for i, mem in enumerate(mem_list, start=1):
        payload[f"F{i}Alias"] = f"{int(mem)}MB"

    start_time = time.time()
    response = sf_client.start_execution(
        stateMachineArn="arn:aws:states:us-east-1:891376968462:stateMachine:ml-pipeline",
        input=json.dumps(payload)
    )
    exec_arn = response["executionArn"]

    # 실행 완료 대기
    while True:
        desc = sf_client.describe_execution(executionArn=exec_arn)
        if desc["status"] in ["SUCCEEDED","FAILED","TIMED_OUT","ABORTED"]:
            break
        time.sleep(1)

    end_time = time.time()
    wall_clock_sec = end_time - start_time

    # default
    final_time = wall_clock_sec
    final_cost = 9999.0

    if desc["status"] == "SUCCEEDED":
        try:
            output_data = json.loads(desc.get("output","{}"))
            # 로그에서 실제 total_time 파싱된 경우 사용
            log_time = output_data.get("log_time", None)   # 가령 "log_time"에 함수별 합산
            if log_time is not None:
                final_time = float(log_time)
            else:
                # 실패 시 fallback = wall-clock
                final_time = wall_clock_sec

            # 람다 비용 직접 계산
            # 예: sum( (mem_i MB/1024) * time_i * $rate ), 여기선 간단히:
            #    cost = final_time * (avg mem / 1024) * 0.0000167
            avg_mem = float(np.mean(mem_list))
            cost_approx = final_time * (avg_mem/1024.0) * 0.0000167
            final_cost = cost_approx

        except:
            final_time = wall_clock_sec
            final_cost = 9999.0
    else:
        # 실패 or timeout 등
        final_time = 999999.0
        final_cost = 9999.0

    return final_time, final_cost


########################################################################
# 2. 문제정의: high-dim input => 2D output (time,cost) => 음수화
########################################################################
class StepFunctionsProblem:
    """
    dim: 함수 개수
    bounds: [128..3008] 각 함수 mem
    """
    def __init__(self, dim=4):
        self.dim = dim
        lb = [128.0]*dim
        ub = [3008.0]*dim
        self.bounds = torch.tensor([lb, ub], dtype=torch.double)

    def __call__(self, X: torch.Tensor):
        """
        X shape: (N, dim)
        -> returns shape (N,2) = [(-time, -cost)]
        """
        out = []
        arr = X.cpu().detach().numpy()
        for i in range(arr.shape[0]):
            mem_list = arr[i,:]
            t, c = run_stepfunctions_workflow(mem_list)
            out.append([-t, -c])
        return torch.tensor(out, dtype=X.dtype, device=X.device)


########################################################################
# 3. 2D MOBO(qNEHVI): hypervolume
########################################################################
def run_mobo(dim=4, n_init=3):
    """
    - input: dim차원(함수 개수)
    - output: ( -time, -cost ) => Pareto 2D
    - 초기점 3개
    - 개선 폭 5% 미만이면 중단
    """

    problem = StepFunctionsProblem(dim=dim)

    # (A) 초기 데이터
    init_x = draw_sobol_samples(
        bounds=problem.bounds, n=n_init, q=1
    ).squeeze(1)  # shape (n_init, dim)
    init_y = problem(init_x)             # shape (n_init,2)

    # (B) 모델
    y_time = init_y[...,0:1]  # -time
    y_cost = init_y[...,1:2]  # -cost
    x_norm = (init_x - problem.bounds[0])/(problem.bounds[1]-problem.bounds[0])

    model_time = SingleTaskGP(x_norm, y_time)
    model_cost = SingleTaskGP(x_norm, y_cost)
    model = ModelListGP(model_time, model_cost)
    mll = SumMarginalLogLikelihood(model.likelihood, model)

    # (C) hypervolume init
    ref_point = torch.tensor([-600000.0, -9999.0], dtype=torch.double)
    bd = DominatedPartitioning(ref_point=ref_point, Y=init_y)
    hv_init = bd.compute_hypervolume().item()
    print(f"[Init] HV={hv_init:.4f}")

    train_x = init_x.clone()
    train_y = init_y.clone()

    best_hv = hv_init

    iteration = 0
    improvement_threshold = 0.05  # 5%
    while True:
        iteration += 1
        # fit
        fit_gpytorch_model(mll)

        # qNEHVI
        sampler = SobolQMCNormalSampler(num_samples=64)
        qnehvi = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point.tolist(),
            X_baseline=train_x,
            sampler=sampler,
            prune_baseline=True,
        )

        # optimize
        dim_ = dim
        bounds_01 = torch.stack([
            torch.zeros(dim_, dtype=torch.double),
            torch.ones(dim_, dtype=torch.double)
        ])
        candidate, _ = optimize_acqf(
            acq_function=qnehvi,
            bounds=bounds_01,
            q=1,
            num_restarts=5,
            raw_samples=32,
            sequential=True
        )
        # unnormalize
        real_x = candidate*(problem.bounds[1]-problem.bounds[0]) + problem.bounds[0]
        new_y = problem(real_x)  # shape(1,2)

        # update train data
        train_x = torch.cat([train_x, real_x], dim=0)
        train_y = torch.cat([train_y, new_y], dim=0)
        x_norm2 = (train_x - problem.bounds[0])/(problem.bounds[1]-problem.bounds[0])
        model_time.set_train_data(x_norm2, train_y[...,0:1], strict=False)
        model_cost.set_train_data(x_norm2, train_y[...,1:2], strict=False)

        # compute HV
        bd = DominatedPartitioning(ref_point=ref_point, Y=train_y)
        hv_val = bd.compute_hypervolume().item()
        print(f"Iter {iteration}, HV={hv_val:.4f}, X={real_x.cpu().numpy()}, Y={new_y.cpu().numpy()}")

        # check improvement ratio
        old_hv = best_hv
        rel_improve = (hv_val - old_hv) / (abs(old_hv)+1e-9)
        if rel_improve < improvement_threshold and iteration>1:
            print(f"Stop criterion triggered: HV improvement < {improvement_threshold*100}%")
            break
        else:
            best_hv = hv_val

    return train_x, train_y


if __name__ == "__main__":
    final_x, final_y = run_mobo(dim=4, n_init=3)
    print("\n=== Done MOBO ===")
    print(f"Data points collected: {final_x.shape[0]}")
    print("Sampled outputs in (-time, -cost):\n", final_y)
