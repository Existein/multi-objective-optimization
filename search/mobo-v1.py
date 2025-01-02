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

# AWS Step Functions 클라이언트
sf_client = boto3.client('stepfunctions', region_name='us-east-1')


###################################################################
# 1. Step Functions 워크플로우 실행 후 (time, cost) 가져오기
###################################################################
def run_stepfunctions_workflow(mem_list):
    """
    mem_list: ex) [m1, m2, m3, m4] (각 함수의 메모리 MB)
    returns: (time_s, cost)
    time_s : 전체 워크플로우 실행 걸린 wall-clock (초)
    cost   : output에서 가져온 total_cost (S3 IO 등 포함)
    """
    payload = {}
    for i, mem in enumerate(mem_list, start=1):
        # Qualifier.$= $.F{i}Alias 구조를 
        # Step Functions JSON에서 바라보도록 하는 예시
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

###################################################################
# 2. 문제 정의 (dim차원 입력 -> 2D 출력(time, cost))
#    음수화해서 (-time, -cost) 반환 => 둘 다 "작을수록 좋음"
###################################################################
class StepFunctionsProblem:
    """
    dim = 함수 개수 (메모리만 최적화)
    메모리 범위 [128..3008] 가정
    """
    def __init__(self, dim):
        self.dim = dim
        lb = [128.0]*dim
        ub = [3008.0]*dim
        self.bounds = torch.tensor([lb, ub], dtype=torch.double)

    def __call__(self, X: torch.Tensor):
        """
        X shape: (N, dim)
        return shape (N,2):  => 2차원 출력 ( -time, -cost )
        """
        out = []
        arr = X.cpu().detach().numpy()
        for i in range(arr.shape[0]):
            mem_list = arr[i,:]
            t, c = run_stepfunctions_workflow(mem_list)
            # time, cost를 최소화 => 음수화
            out.append([-t, -c])
        return torch.tensor(out, dtype=X.dtype, device=X.device)


###################################################################
# 3. qNoisyExpectedHypervolumeImprovement (2D MOBO)
###################################################################
def run_mobo(dim, n_init, n_iter):
    """
    2D MOBO:
    - input dim = #함수 (각 함수 메모리)
    - output 2D = ( -time, -cost )
    - hypervolume ref point 예: [ -60000, -1.0 ] (time=60000s, cost=1$ worst)
    """
    problem = StepFunctionsProblem(dim=dim)

    # (1) 초기 데이터: Sobol
    init_x = draw_sobol_samples(
        bounds=problem.bounds, n=n_init, q=1
    ).squeeze(1)  # shape (n_init, dim)
    init_y = problem(init_x)            # shape (n_init, 2)

    # (2) 모델: 2개 SingleTaskGP => ModelListGP
    #    y0= -time, y1= -cost
    y_time = init_y[...,0:1]  # shape (n_init,1)
    y_cost = init_y[...,1:2]  # shape (n_init,1)

    # input 정규화
    x_norm = (init_x - problem.bounds[0])/(problem.bounds[1]-problem.bounds[0])
    model_time = SingleTaskGP(x_norm, y_time)
    model_cost = SingleTaskGP(x_norm, y_cost)
    model = ModelListGP(model_time, model_cost)
    mll = SumMarginalLogLikelihood(model.likelihood, model)

    # (3) 초기 하이퍼볼륨 계산
    #     ref_point e.g. [-999999, -9999], 혹은 [-60000, -1.0] 등
    ref_point = torch.tensor([-60000.0, -10.0], dtype=torch.double)
    bd = DominatedPartitioning(ref_point=ref_point, Y=init_y)
    hv_init = bd.compute_hypervolume().item()
    print(f"[Init] Hypervolume={hv_init:.4f}")

    train_x = init_x.clone()
    train_y = init_y.clone()

    for it in range(1, n_iter+1):
        # (4) 모델 학습
        fit_gpytorch_model(mll)

        # (5) qNoisyExpectedHypervolumeImprovement
        sampler = SobolQMCNormalSampler(num_samples=64)
        qnehvi = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point.tolist(),
            X_baseline=train_x,
            sampler=sampler,
            prune_baseline=True,
        )

        # (6) optimize_acqf
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
        # (7) 관측
        new_y = problem(real_x)  # shape(1,2)

        # update train data
        train_x = torch.cat([train_x, real_x], dim=0)
        train_y = torch.cat([train_y, new_y], dim=0)
        x_norm2 = (train_x - problem.bounds[0])/(problem.bounds[1]-problem.bounds[0])
        model_time.set_train_data(x_norm2, train_y[...,0:1], strict=False)
        model_cost.set_train_data(x_norm2, train_y[...,1:2], strict=False)

        # (8) hypervolume
        bd = DominatedPartitioning(ref_point=ref_point, Y=train_y)
        hv_val = bd.compute_hypervolume().item()
        print(f"Iter {it}, HV={hv_val:.4f}, X={real_x.cpu().numpy()}, Y={new_y.cpu().numpy()}")

    return train_x, train_y


if __name__ == "__main__":
    final_x, final_y = run_mobo(dim=4, n_init=3, n_iter=20)
    print("\n=== Done MOBO ===")
    print(f"Final data points: {final_x.shape[0]}")
    print("Sampled outputs ( -time, -cost ):\n", final_y)    
