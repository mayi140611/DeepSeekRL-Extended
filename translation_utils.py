from collections import Counter
import math

def get_ngrams(segment, n):
    """返回句子的n-gram列表，每个元素是一个元组。"""
    return [tuple(segment[i:i+n]) for i in range(len(segment)-n+1)]

def compute_bleu(references, candidates, max_n=4, weights=[0.25, 0.25, 0.25, 0.25]):
    """
    计算BLEU分数，支持多个参考翻译。
    :param references: 列表的列表，每个元素是对应候选翻译的参考翻译列表。例如: [[ref1, ref2], [ref3, ref4]]
    :param candidates: 列表的列表，候选翻译。例如: [[cand1], [cand2]]
    :param max_n: 最大n-gram阶数（默认为4)
    :param weights: 各阶n-gram的权重（默认等权，总和为1)
    :return BLEU-4分数（0~1）
    """
    assert max_n == len(weights), "权重数量应与max_n相等"
    
    # 统计各n-gram匹配数和总候选n-gram数
    p_ns = []
    for n in range(1, max_n+1):
        total_matches = 0
        total_candidate = 0
        
        # 逐句处理
        for candidate, ref_list in zip(candidates, references):
            # 当前候选句子的n-gram数量
            candidate_grams = get_ngrams(candidate, n)
            total_candidate += len(candidate_grams)
            if len(candidate_grams) == 0:
                continue  # 跳过无法生成n-gram的情况
            
            # 统计候选中每个n-gram的出现次数
            candidate_counts = Counter(candidate_grams)
            
            # 统计所有参考中n-gram的最大次数
            max_ref_counts = {}
            for ref in ref_list:
                ref_grams = get_ngrams(ref, n)
                ref_counts = Counter(ref_grams)
                for gram, count in ref_counts.items():
                    max_ref_counts[gram] = max(count, max_ref_counts.get(gram, 0))
                
            # 剪裁后匹配数目
            clipped_matches = 0
            for gram, count in candidate_counts.items():
                clipped_matches += min(count, max_ref_counts.get(gram, 0))
            total_matches += clipped_matches
        
        # 计算精度
        p_n = total_matches / total_candidate if total_candidate != 0 else 0
        p_ns.append(p_n)
    
    # ----------------------------------
    # 计算Brevity Penalty（BP）
    c = sum(len(cand) for cand in candidates)   # 候选总长度
    r = 0   # 参考有效总长度
    for cand, ref_list in zip(candidates, references):
        # 找到每个候选对应的最优参考长度（最接近候选长度的参考）
        ref_lens = [len(ref) for ref in ref_list]
        closest_ref_len = min(ref_lens, key=lambda x: (abs(x - len(cand)), x))
        r += closest_ref_len
    
    if c == 0:
        return 0.0  # 避免除零
    
    bp = math.exp(1 - r / c) if c <= r else 1.0
    
    # 计算几何加权平均
    s = 0.0
    for p_n, weight in zip(p_ns, weights):
        if p_n == 0:
            return 0.0
        s += weight * math.log(p_n)
    bleu = bp * math.exp(s)
    
    return bleu