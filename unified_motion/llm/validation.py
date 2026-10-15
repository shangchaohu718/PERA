"""Local admission evidence for LLM proposals; model opinions cannot approve themselves."""
import hashlib
import math
import statistics

from .advisor import fingerprint, validate_changes, validate_curriculum_evidence

TRACKING_METRICS = ('root_position_error_m', 'root_orientation_error_rad',
                    'root_linear_velocity_error_mps', 'keybody_linear_velocity_rmse_mps')


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def compare_trials(candidate, baseline_reports, candidate_reports, *, max_cost_ratio=1.25,
                   primary_metric='root_position_error_m'):
    """Require matched frozen evals from equal-step forks and >=3 evaluation seeds.

    The decision is an empirical admission check, not a statistical guarantee.
    A changed shaped reward is deliberately excluded from the decision.
    """
    if not math.isfinite(max_cost_ratio) or max_cost_ratio < 1:
        raise ValueError('invalid cost ratio budget')
    if primary_metric not in TRACKING_METRICS:
        raise ValueError('primary metric must be a fixed tracking metric')
    if (candidate.get('schema') != 1 or candidate.get('status') != 'unvalidated_candidate'
            or candidate.get('settings_sha256') != fingerprint(candidate.get('settings_before'))
            or validate_changes(candidate['settings_before'], candidate['changes']) != candidate['settings_after']):
        raise ValueError('invalid candidate provenance')
    evidence = candidate.get('evidence')
    if not isinstance(evidence, dict) or fingerprint(evidence) != candidate.get('evidence_sha256'):
        raise ValueError('invalid candidate evidence')
    validate_curriculum_evidence(candidate['settings_before'], candidate['changes'], evidence)
    def index(reports):
        result = {}
        for report in reports:
            protocol = report.get('evaluation_protocol')
            if not isinstance(protocol, dict) or report.get('evaluation_protocol_sha256') != fingerprint(protocol):
                raise ValueError('missing or inconsistent evaluation protocol')
            key = fingerprint(protocol)
            if key in result:
                raise ValueError('duplicate evaluation case')
            if (report.get('weights_unchanged') is not True or report.get('deterministic') is not True
                    or report.get('backend') != 'mjlab_cuda' or protocol.get('ablation') != 'full'):
                raise ValueError('admission requires real, frozen full-policy evaluation')
            result[key] = report
        return result
    baseline, trial = index(baseline_reports), index(candidate_reports)
    if not baseline or baseline.keys() != trial.keys():
        raise ValueError('evaluation cases, seeds and budgets must match')
    seeds = {r['evaluation_protocol'].get('seed') for r in baseline.values()}
    if len(seeds) < 3 or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('at least three distinct evaluation seeds are required')
    evidence, accepted, outcomes_improved = [], True, False
    paired_errors = {name: [] for name in TRACKING_METRICS}
    for key, base in baseline.items():
        new = trial[key]
        training_protocol = base.get('training_protocol')
        if (not isinstance(training_protocol, dict) or not training_protocol
                or training_protocol != new.get('training_protocol')):
            raise ValueError('matched training motions, task and batch configuration are required')
        source = base.get('training_source_checkpoint_sha256')
        if not isinstance(source, str) or len(source) != 64 or source != new.get('training_source_checkpoint_sha256'):
            raise ValueError('baseline/candidate must start from the same saved checkpoint')
        if (type(base.get('training_seed')) is not int or base['training_seed'] != new.get('training_seed')
                or new.get('trial_adjustment_sha256') != fingerprint(candidate)):
            raise ValueError('candidate or training seed provenance differs')
        if base.get('trial_adjustment_sha256') == fingerprint(candidate):
            raise ValueError('baseline cannot already contain the candidate adjustment')
        before_cost, after_cost = base.get('training_cost') or {}, new.get('training_cost') or {}
        if (type(before_cost.get('transitions')) is not int or before_cost['transitions'] <= 0
                or before_cost['transitions'] != after_cost.get('transitions')):
            raise ValueError('matched positive training transition counts are required')
        seconds = (before_cost.get('compute_seconds'), after_cost.get('compute_seconds'))
        if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in seconds):
            raise ValueError('measured training cost is required')
        cost_ratio = seconds[1]/seconds[0]
        passed = cost_ratio <= max_cost_ratio
        for name in TRACKING_METRICS:
            old, value = (r.get('tracking', {}).get(name) for r in (base, new))
            if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in (old, value)):
                raise ValueError('finite fixed tracking errors are required')
            passed &= value <= old*1.02+1e-8
            paired_errors[name].append((old, value))
        for name in ('physical_failures', 'reference_completions'):
            old, value = base.get(name), new.get(name)
            if any(type(x) is not int or x < 0 for x in (old, value)):
                raise ValueError('completion/failure counts are required')
            passed &= value <= old if name == 'physical_failures' else value >= old
            outcomes_improved |= value < old if name == 'physical_failures' else value > old
        accepted &= bool(passed)
        evidence.append(dict(protocol_sha256=key, passed=bool(passed), cost_ratio=cost_ratio))
    aggregate = {}
    for name, pairs in paired_errors.items():
        before = statistics.mean(pair[0] for pair in pairs)
        after = statistics.mean(pair[1] for pair in pairs)
        aggregate[name] = dict(baseline_mean=before, candidate_mean=after,
            relative_change=(after-before)/max(before, 1e-8), nonregression=after <= before+1e-8)
    # No tradeoff across different physical units: every metric must preserve
    # its paired-case mean. An isolated winning metric/seed cannot pass a trial.
    nonregression = all(item['nonregression'] for item in aggregate.values())
    primary = aggregate[primary_metric]
    primary_improved = primary['candidate_mean'] < primary['baseline_mean']*.98
    return dict(schema=2, accepted=bool(accepted and nonregression and (primary_improved or outcomes_improved)),
        candidate_sha256=fingerprint(candidate), primary_metric=primary_metric, aggregate_tracking=aggregate,
        aggregate_nonregression=nonregression, primary_improved=primary_improved,
        outcomes_improved=outcomes_improved,
        cases=evidence, evaluation_seeds=sorted(seeds), max_cost_ratio=max_cost_ratio,
        explanation='Each case: tracking tolerance 2%, outcomes/cost guarded. Each tracking metric: paired-case mean cannot worsen. Requires >2% mean improvement in the preselected primary metric or better outcomes. Not statistical significance.',
        deployment_changed=False)
