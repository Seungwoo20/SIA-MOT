# %%
import numpy as np
from scipy.optimize import linear_sum_assignment # 선형 할당 문제를 풀기 위한 라이브러리(Linear Assignment Problem)
from ._base_metric import _BaseMetric
from . import _timing
from . import utils

class CLEAR(_BaseMetric): # _BaseMetric를 상속하는 class(지표에 대한 기본 클래스)
    """Class which implements the CLEAR metrics"""

    @staticmethod
    def get_default_config(): # 지표의 기본 구성 값이 포함된 딕셔너리를 반환
        """Default class config values"""
        default_config = {
            'THRESHOLD': 0.5,  # True Positive에 대한 threshold
            'PRINT_CONFIG': True,  # Whether to print the config information on init. Default: False.
        }
        return default_config

    def __init__(self, config=None): # CLEAR 객체를 초기화
        super().__init__
        main_integer_fields = ['CLR_TP', 'CLR_FN', 'CLR_FP', 'IDSW', 'MT', 'PT', 'ML', 'Frag']
        # CLR_TP - True Positive(올바르게 매칭된 추적 결과의 수)
        # CLR_FN - False Negative(매칭되지 않은 실제 객체의 수)
        # CLR_FP - False Positive(잘못 매칭된 추적 결과의 수)
        # MT(Most Tracked) - 대부분의 시간 동안 정확히 추적된 객체의 수
        # PT(Partially Tracked) - 일부 시간 동안 추적된 객체의 수
        # ML(Mostly Lost) - 대부분의 시간 동안 추적을 실패한 객체의 수
        # Frag(Fragments) - 중간에 끊어진 추적 경로의 개수
        
        extra_integer_fields = ['CLR_Frames']
        # CLR_Frames - 평가 대상 시퀀스의 프레임 수
        
        self.integer_fields = main_integer_fields + extra_integer_fields
        
        main_float_fields = ['MOTA', 'MOTP', 'MODA', 'CLR_Re', 'CLR_Pr', 'MTR', 'PTR', 'MLR', 'sMOTA'] # 주요 부동 소수점 필드 목록
        # MOTA(Multiple Object Tracking Accuracy) - 다중 객체 추적 정확도
        # MOTP (Multiple Object Tracking Precision) - 다중 객체 추정 정밀도
        # MODA (Multiple Object Detection Accuracy) - 다둥 객체 감지 정확도
        # CLR_Re (CLEAR Recall) - CLEAR 재현율
        # CLR_Pr (CLEAR Precision) - CLEAR 정밀도
        # MTR (Multiple Object Tracking Ratio) - 다중 객체 추적 비율
        # PTR (Partial Tracking Ratio) - 부분 추적 비율
        # MLR (Mostly Lost Ratio) - 대부분 손실 비율
        # sMOTA (Scaled MOTA) - 조정된 MOTA
        
        extra_float_fields = ['CLR_F1', 'FP_per_frame', 'MOTAL', 'MOTP_sum'] # 추가 부동 소수점 필드 목록
        # CLR_F1 (CLEAR F1 Score) - CLEAR F1 점수
        # FP_per_frame (False Positives per Frame) - 프레임 당 거짓 양성 수
        # MOTAL (Multiple Object Tracking Accuracy with Log) - 로그를 적용한 다중 객체 추적 정확도
        # MOTP_sum (Sum of MOTP) - MOTP의 합계
    
        self.float_fields = main_float_fields + extra_float_fields
        self.fields = self.float_fields + self.integer_fields
        
        self.summed_fields = self.integer_fields + ['MOTP_sum']
        self.summary_fields = main_float_fields + main_integer_fields

        # Configuration options:
        self.config = utils.init_config(config, self.get_default_config(), self.get_name())
        self.threshold = float(self.config['THRESHOLD'])


    @_timing.time
    def eval_sequence(self, data): # 단일 시퀀스에 대한 CLEAR 지표를 계산
        """Calculates CLEAR metrics for one sequence"""
        
        # data -> groundtruth와 예측된 객체 추적에 대한 정보
        
        # result 초기화
        res = {}
        for field in self.fields:
            res[field] = 0

        # tracker나 gt sentence가 비었으면 result를 즉시 return
        if data['num_tracker_dets'] == 0:
            res['CLR_FN'] = data['num_gt_dets']
            res['ML'] = data['num_gt_ids']
            res['MLR'] = 1.0
            return res
        
        if data['num_gt_dets'] == 0:
            res['CLR_FP'] = data['num_tracker_dets']
            res['MLR'] = 1.0
            return res

        # 전역 연관 관련 변수를 초기화
        num_gt_ids = data['num_gt_dets'] # Ground Truth 식별자의 총 수
        
        gt_id_count = np.zeros(num_gt_ids)  # 관심 대상 식별자별로 MT, ML, PT 계산을 위한 배열
                                            # 얼마나 개별 관심 대상 식별자를 얼마나 잘 추적했는지, 
                                            # 얼마나 많은 프레임에서 추적되었는지,
                                            # 얼마나 많은 프레임에서 손실되었는지 추적하는데 사용
                                            
        gt_matched_count = np.zeros(num_gt_ids)  # 관심 대상 식별자별로 MT, ML, PT 계산을 위한 배열
                                                 # 실제로 추적된 관심 대상 식별자의 수를 추적하는데 사용
                                                 
        gt_frag_count = np.zeros(num_gt_ids)  # 관심 대상 식별자별로 Frag(Fragmentation) 계산하기 위한 배열
                                              # 얼마나 많은 프레임에서 손실되었다가 다시 추적되는 관심 대상 식별자를 추적하는데 사용

        # Note that IDSWs are counted based on the last time each gt_id was present (any number of frames previously),
        # 각 gt_id가 이전에 언제 존재했는지(이전의 임의의 프레임 수에 걸쳐)에 따라 IDSW(Identity Switch)가 계산
        # but are only used in matching to continue current tracks based on the gt_id in the single previous timestep.
        # 그러나 IDSW는 매칭에만 사용되며, 현재 트랙을 이전 단일 시간 단계의 gt_id를 기반으로 유지하기 위해 사용
        
        prev_tracker_id = np.nan * np.zeros(num_gt_ids)  # IDSW 계산을 위한 배열
        prev_timestep_tracker_id = np.nan * np.zeros(num_gt_ids)  # 각 gt_id에 대해 이전 시간 단계에서 매칭되었던 tracker_id를 기록(없는 경우 NaN)

        # 타임스팀프에 대한 각각의 점수 계산
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            # Deal with the case that there are no gt_det/tracker_det in a timestep.
            # gt_det 또는 tracker_det이 특정 타임스탬프에 없는 경우를 처리
            # gt_ids_t -> 해당 시간 단계에서의 ground truth detection의 식별자(ID)
            
            # FP 계산
            
            if len(gt_ids_t) == 0:
                res['CLR_FP'] += len(tracker_ids_t)
                continue
            
            # FN 계산
            if len(tracker_ids_t) == 0:
                res['CLR_FN'] += len(gt_ids_t)
                gt_id_count[gt_ids_t-1] += 1
                continue

            # Calc score matrix to first minimise IDSWs from previous frame, and then maximise MOTP secondarily
            # 이전 프레임에서 IDSW(ID Switches)를 최소화하기 위해 점수 행렬을 계산하고, 그 다음으로 MOTP(Mean Overlap Ratio of True Positives)를 최대화하기 위해 계산
            
            similarity = data['similarity_scores'][t]
            
            # score_mat -> 점수 행렬
            
            score_mat = (tracker_ids_t[np.newaxis, :] == prev_timestep_tracker_id[gt_ids_t[:, np.newaxis]-1])
            score_mat = 1000 * score_mat + similarity
            score_mat[similarity < self.threshold - np.finfo('float').eps] = 0

            # Hungarian algorithm to find best matches
            # 헝가리안 알고리즘 사용
            
            match_rows, match_cols = linear_sum_assignment(-score_mat)
            actually_matched_mask = score_mat[match_rows, match_cols] > 0 + np.finfo('float').eps
            match_rows = match_rows[actually_matched_mask]
            match_cols = match_cols[actually_matched_mask]

            matched_gt_ids = gt_ids_t[match_rows]
            matched_tracker_ids = tracker_ids_t[match_cols]

            # Calc IDSW for MOTA
            # MOTA에 쓰일 IDSW 계산
            
            prev_matched_tracker_ids = prev_tracker_id[matched_gt_ids]
            is_idsw = (np.logical_not(np.isnan(prev_matched_tracker_ids))) & (
                np.not_equal(matched_tracker_ids, prev_matched_tracker_ids))
            res['IDSW'] += np.sum(is_idsw)

            # Update counters for MT/ML/PT/Frag and record for IDSW/Frag for next timestep
            # 다음 타임스탬프를 위해 MT/ML/PT/Frag 및 IDSW/Frag을(를) 업데이트하고 기록
            
            gt_id_count[gt_ids_t-1] += 1
            gt_matched_count[matched_gt_ids] += 1
            not_previously_tracked = np.isnan(prev_timestep_tracker_id)
            prev_tracker_id[matched_gt_ids] = matched_tracker_ids
            prev_timestep_tracker_id[:] = np.nan
            prev_timestep_tracker_id[matched_gt_ids] = matched_tracker_ids
            currently_tracked = np.logical_not(np.isnan(prev_timestep_tracker_id))
            gt_frag_count += np.logical_and(not_previously_tracked, currently_tracked)
            # Calculate and accumulate basic statistics
            # 기본 통계를 계산하고 누적
            
            num_matches = len(matched_gt_ids)
            res['CLR_TP'] += num_matches
            res['CLR_FN'] += len(gt_ids_t) - num_matches
            res['CLR_FP'] += len(tracker_ids_t) - num_matches
            if num_matches > 0:
                res['MOTP_sum'] += sum(similarity[match_rows, match_cols])

        # Calculate MT/ML/PT/Frag/MOTP
        # MT/ML/PT/Frag/MOTP 계산
        tracked_ratio = gt_matched_count[gt_id_count > 0] / gt_id_count[gt_id_count > 0]
        res['MT'] = np.sum(np.greater(tracked_ratio, 0.8))
        res['PT'] = np.sum(np.greater_equal(tracked_ratio, 0.2)) - res['MT']
        res['ML'] = num_gt_ids - res['MT'] - res['PT']
        res['Frag'] = np.sum(np.subtract(gt_frag_count[gt_frag_count > 0], 1))
        res['MOTP'] = res['MOTP_sum'] / np.maximum(1.0, res['CLR_TP'])

        res['CLR_Frames'] = data['num_timesteps']

        # Calculate final CLEAR scores
        # 최종 CLEAR 점수 계산
        res = self._compute_final_fields(res)
        return res
    
    # 모든 시퀀스에 대해 Metrics를 결합
    # input : all_res(딕셔너리) - 각 시퀀스에 대한 Metrics
    # 모든 시퀀스에 대한 metrics)를 결합하여 하나의 결과를 생성

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        for field in self.summed_fields:
            res[field] = self._combine_sum(all_res, field)
        res = self._compute_final_fields(res)
        return res
    
    # 모든 클래스에 대해 Metrics를 결합
    # input : all_res(딕셔너리) - 각 클래스에 대한 Metrics
    # 각 클래스에 대한 metrics를 결합하여 하나의 결과를 생성

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the detection values"""
        res = {}
        for field in self.summed_fields:
            res[field] = self._combine_sum(all_res, field)
        res = self._compute_final_fields(res)
        return res
    
    #  모든 클래스에 대해 Metrics를 결합
    # input : all_res(딕셔너리), ignore_empty_classes(bool값)

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=False):
        """Combines metrics across all classes by averaging over the class values.
        If 'ignore_empty_classes' is True, then it only sums over classes with at least one gt or predicted detection.
        """
        res = {}
        # 1. 정수 필드
        for field in self.integer_fields:
            
            # 빈 클래스를 무시할 경우
            # 적어도 하나의 ground truth 또는 예측된 detection이 있는 클래스만을 고려하여 값을 합산
            # all_res에서 CLR_TP, CLR_FN, CLR_FP 중 하나라도 값이 0보다 큰 클래스들만 선택하여 처리
            if ignore_empty_classes:
                res[field] = self._combine_sum(
                    {k: v for k, v in all_res.items() if v['CLR_TP'] + v['CLR_FN'] + v['CLR_FP'] > 0}, field)
                
            # 빈 클래스를 무시하지 않을 경우
            # 모든 값을 합산
            else:
                res[field] = self._combine_sum({k: v for k, v in all_res.items()}, field)
        
        # 2. 실수 필드        
        for field in self.float_fields:
            
            # 빈 클래스를 무시할 경우
            # 적어도 하나의 ground truth 또는 예측된 detection이 있는 클래스만을 고려하여 값을 합산
            # all_res에서 CLR_TP, CLR_FN, CLR_FP 중 하나라도 값이 0보다 큰 클래스들만 선택하여 처리
            if ignore_empty_classes:
                res[field] = np.mean(
                    [v[field] for v in all_res.values() if v['CLR_TP'] + v['CLR_FN'] + v['CLR_FP'] > 0], axis=0)
                
            # 빈 클래스를 무시하지 않을 경우
            # 모든 값을 합산
            else:
                res[field] = np.mean([v[field] for v in all_res.values()], axis=0)
        return res

    @staticmethod
    def _compute_final_fields(res):
        """Calculate sub-metric ('field') values which only depend on other sub-metric values.
        This function is used both for both per-sequence calculation, and in combining values across sequences.
        """
        num_gt_ids = res['MT'] + res['ML'] + res['PT']
        res['MTR'] = res['MT'] / np.maximum(1.0, num_gt_ids)
        res['MLR'] = res['ML'] / np.maximum(1.0, num_gt_ids)
        res['PTR'] = res['PT'] / np.maximum(1.0, num_gt_ids)
        res['CLR_Re'] = res['CLR_TP'] / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
        res['CLR_Pr'] = res['CLR_TP'] / np.maximum(1.0, res['CLR_TP'] + res['CLR_FP'])
        res['MODA'] = (res['CLR_TP'] - res['CLR_FP']) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
        res['MOTA'] = (res['CLR_TP'] - res['CLR_FP'] - res['IDSW']) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
        res['MOTP'] = res['MOTP_sum'] / np.maximum(1.0, res['CLR_TP'])
        res['sMOTA'] = (res['MOTP_sum'] - res['CLR_FP'] - res['IDSW']) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])

        res['CLR_F1'] = res['CLR_TP'] / np.maximum(1.0, res['CLR_TP'] + 0.5*res['CLR_FN'] + 0.5*res['CLR_FP'])
        res['FP_per_frame'] = res['CLR_FP'] / np.maximum(1.0, res['CLR_Frames'])
        safe_log_idsw = np.log10(res['IDSW']) if res['IDSW'] > 0 else res['IDSW']
        res['MOTAL'] = (res['CLR_TP'] - res['CLR_FP'] - safe_log_idsw) / np.maximum(1.0, res['CLR_TP'] + res['CLR_FN'])
        return res

# %%
"""
### scipy.optimize.linear_sum_assignment

- 선형 할당 문제(Linear Assignment Problem)를 풀기 위한 함수
- 선형 할당 문제는 일반적으로 비용 행렬 또는 유사도 행렬이 주어지며, 첫 번째 집합의 요소를 두 번째 집합의 요소와 매핑시키는 방식으로 해결
- 헝가리안 알고리즘(Hungarian algorithm)을 사용하여 선형 할당 문제를 해결
- 시간 복잡도가 O(n^3)이며, 일반적으로 작은 크기의 할당 문제에 사용

- 인수
    - cost_matrix : 비용 행렬 또는 유사도 행렬로, 첫 번째 집합과 두 번째 집합 간의 비용 또는 유사도를 나타냅니다. 일반적으로 NxM 크기의 2D 배열로 표현
    - method : 헝가리안 알고리즘의 최적화 방법을 선택합니다. 기본값은 "auction"이며, 다른 옵션으로 "path"와 "dense

- 최적의 매칭을 나타내는 두 개의 배열을 반환
    - row_ind : 첫 번째 집합의 요소와 매칭된 두 번째 집합의 요소의 인덱스 배열
    - col_ind : 두 번째 집합의 요소와 매칭된 첫 번째 집합의 요소의 인덱스 배열
"""

# %%
"""
###  ._base_metric._BaseMetric

- CLEAR 클래스에서 상속한 _BaseMetric이라는 모듈
- CLEAR 클래스의 기본 지표 클래스로 사용되며, 지표 계산에 필요한 공통 기능과 구조를 정의

- 기능
    - get_name : 지표의 이름을 반환(하위 클래스에서 구현되어야함)
    - get_fields : 지표에서 반환되는 필드(메트릭)의 이름 목록을 반환(하위 클래스에서 구현되어야함)
    - eval_sequence : 하위 클래스에서 구현되어야 하는 메서드로, 단일 시퀀스에 대한 지표를 계산, 시퀀스에 대한 입력 데이터를 받아 필요한 계산을 수행하고, 결과를 딕셔너리 형태로 반환
    - combine_sequences : 모든 시퀀스의 지표를 결합하여 전체 시퀀스에 대한 종합 지표를 계산하는 메서드입니다. 단일 시퀀스 결과의 리스트를 입력으로 받아 필요한 필드를 합산하여 결과를 반환
"""

# %%
"""
### INPUT이 될 - data(배열)가 가져야할 정보
- num_tracker_dets: 추적기에서 감지한 객체의 수
- num_gt_dets: 실제 객체의 수
- gt_ids: 각 타임스텝에서의 실제 객체의 식별자(ID) 배열
- tracker_ids: 각 타임스텝에서 추적기가 예측한 객체의 식별자(ID) 배열
- similarity_scores: 타임스텝별 유사도 점수 행렬
"""

