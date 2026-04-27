"""
Deep audit script — runs against a realistic fake DebateState
and prints every reward / trajectory output for manual inspection.
"""
import json
import sys

sys.path.insert(0, '.')

from backend.engine.state import DebateState
from backend.rl.reward import compute_reward, compute_per_agent_rewards
from backend.rl.trajectory import (
    log_trajectory, get_trajectory, get_stats, get_trajectory_list
)

fake_state: DebateState = {
    'user_query': 'Is there gender bias in our hiring dataset?',
    'dataset_csv': 'age,gender,caste,hired\n28,Male,General,1\n25,Female,SC,0',
    'dataset_summary': 'Shape: 50 rows x 7 cols. Male 80% hired, Female 40% hired.',
    'use_case': 'job_hiring',
    'protected_attributes': ['gender', 'caste'],
    'target_column': 'hired',
    'model_backend': 'gemini-2.5-flash',
    'fairness_metrics': {
        'total_rows': 50,
        'overall_positive_rate': 0.6,
        'gender[Male]_selection_rate': 0.8,
        'gender[Female]_selection_rate': 0.4,
        'gender_demographic_parity_difference': -0.4,
        'gender_disparate_impact_violated': True,
        'gender_equalized_odds_difference': 0.35,
        'gender[Male]_accuracy': 1.0,
        'gender[Female]_accuracy': 1.0,
        'gender_accuracy_disparity': 0.0,
        'gender_disparate_impact_ratio': 0.5,
        'gender_4_5ths_rule_violated': True,
    },
    'debate_messages': [
        {
            'agent': 'data_statistician',
            'role': 'Data Statistician',
            'content': (
                'Gender disparity: male 80% vs female 40%. '
                'Demographic parity difference of -0.4 exceeds the 0.1 threshold. '
                'The 4/5ths rule violated: disparate_impact_ratio=0.5. Confidence: High. '
                'Question for fairness auditor: which formal definition applies?'
            ),
            'round': 0,
        },
        {
            'agent': 'fairness_auditor',
            'role': 'Fairness Auditor',
            'content': (
                'As the statistician noted, disparate impact is clearly violated. '
                'Statistical parity difference -0.4 >> 0.1 threshold. '
                'Severity: Critical. Equalized odds difference 0.35 also violated. '
                'Trade-off: cannot satisfy both statistical parity and equalized odds. '
                'Priority: statistical parity for this job_hiring use case.'
            ),
            'round': 0,
        },
        {
            'agent': 'domain_expert',
            'role': 'Domain Expert',
            'content': (
                'In job_hiring, a 40% female rate vs 80% male rate signals systemic exclusion. '
                'years_experience may be a proxy variable for gender bias. '
                'interview_score warrants scrutiny. '
                'Real-world impact: female candidates face 50% lower chance of positive outcome.'
            ),
            'round': 0,
        },
        {
            'agent': 'bias_adversary',
            'role': 'Bias Adversary',
            'content': (
                'I challenge the statistician conclusion. Sample size is only 50 rows. '
                'interview_score differences could reflect merit. '
                'However the fairness auditor and statistician evidence is compelling. '
                'I concede partially but flag the small sample size as a methodology concern.'
            ),
            'round': 0,
        },
        {
            'agent': 'ethical_reviewer',
            'role': 'Ethical & Legal Reviewer',
            'content': (
                'This dataset violates DPDP Act 2023 on discrimination by sex. '
                'EEOC Title VII disparate impact doctrine applies. '
                'GDPR Article 22 requires safeguards on automated decisions. '
                'Article 15 of the Indian Constitution prohibits this discrimination. '
                'Intersectional bias: female SC candidates face double discrimination. '
                'Immediate disclosure and legal review required.'
            ),
            'round': 0,
        },
        {
            'agent': 'data_statistician',
            'role': 'Data Statistician',
            'content': (
                'Responding to the adversary: even at 50 rows, '
                'chi-square test shows p<0.01. The magnitude of 40% vs 80% '
                'overcomes the sample size limitation raised by the adversary.'
            ),
            'round': 1,
        },
        {
            'agent': 'bias_adversary',
            'role': 'Bias Adversary',
            'content': (
                'I accept the statistician chi-square argument. Evidence is overwhelming. '
                'I withdraw the sample-size objection. The domain expert proxy variable '
                'point is well-taken. Remaining concern: reweighing mitigation could destabilise the model.'
            ),
            'round': 1,
        },
    ],
    'current_round': 2,
    'max_rounds': 2,
    'final_report': {
        'bias_score': 78,
        'severity': 'High',
        'summary': 'Severe gender bias detected in hiring dataset. Female candidates face 50% lower selection rate.',
        'flagged_issues': [
            {
                'issue': 'Gender Disparate Impact',
                'description': 'DI ratio 0.5 — violates 4/5ths rule',
                'evidence': 'gender_disparate_impact_ratio: 0.5',
                'severity': 'Critical',
            },
            {
                'issue': 'Statistical Parity Violation',
                'description': 'Difference -0.4 far exceeds 0.1 threshold',
                'evidence': 'gender_demographic_parity_difference: -0.4',
                'severity': 'Critical',
            },
            {
                'issue': 'DPDP Act Violation',
                'description': 'Gender discrimination under Indian data protection law',
                'evidence': 'Article 15 applicable',
                'severity': 'High',
            },
        ],
        'mitigation_steps': [
            {
                'step': 'Apply reweighing pre-processing',
                'priority': 'Immediate',
                'description': (
                    'Use Fairlearn reweighing to rebalance training data before model '
                    'training to eliminate gender-based selection disparities completely.'
                ),
            },
            {
                'step': 'Audit interview scoring rubric',
                'priority': 'Short-term',
                'description': (
                    'Review interview scoring criteria for gender-correlated bias '
                    'and remove or neutralise biased components from the evaluation pipeline.'
                ),
            },
        ],
        'agent_consensus': {
            'agreed_on': ['Gender bias is critical', 'DI ratio violated'],
            'disputed': ['Sample size adequacy — resolved in round 2 via chi-square test'],
        },
        'legal_risk': 'High',
        'confidence': 'High',
    },
    'error': '',
}

# --- 1. Global reward ---
print('=== GLOBAL REWARD ===')
gr = compute_reward(fake_state)
for k, v in gr.items():
    print(f'  {k}: {v}')

# --- 2. Per-agent rewards ---
print('\n=== PER-AGENT REWARDS ===')
pa = compute_per_agent_rewards(fake_state)
for agent_id, scores in pa.items():
    total = scores.get('total', '?')
    print(f'  [{agent_id}]  total={total}')
    for k, v in scores.items():
        if k != 'total':
            print(f'    {k}: {v}')

# --- 3. Log a trajectory and read it back ---
print('\n=== TRAJECTORY LOG + READBACK ===')
debate_id = log_trajectory(fake_state)
print(f'  Logged debate_id: {debate_id}')

rec = get_trajectory(debate_id)
assert rec is not None, 'get_trajectory returned None for a just-logged debate!'
print(f'  Read back: debate_id={rec["debate_id"]}')
print(f'  Reward in record: {rec.get("reward", {}).get("total")}')
print(f'  Per-agent keys: {list(rec.get("per_agent_rewards", {}).keys())}')
assert rec.get('marti_format'), 'marti_format is empty!'
print(f'  MARTI turns: {len(rec["marti_format"])}')

# --- 4. Stats ---
print('\n=== STATS ===')
stats = get_stats()
print(json.dumps(stats, indent=2))

# --- 5. List ---
print('\n=== TRAJECTORY LIST ===')
lst = get_trajectory_list(limit=5)
print(f'  {len(lst)} item(s) returned')
if lst:
    item = lst[0]
    expected_keys = {'debate_id','timestamp','use_case','model_backend','bias_score','severity','total_reward'}
    missing = expected_keys - set(item.keys())
    if missing:
        print(f'  WARNING: list item missing keys: {missing}')
    else:
        print(f'  List item keys OK: {list(item.keys())}')

print('\nALL ASSERTIONS PASSED.')
