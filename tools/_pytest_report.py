"""pytest报告插件；由evaluate.py在独立进程加载，避免漏收函数式测试。

不导入仿真或网络，只记录pytest已经产生的结果。子用例单列，不重复计总数。
"""
import json
import os
from pathlib import Path

_items = []
_records = {}
_errors = []
_subtests = dict(passed=0, failed=0, skipped=0)


def pytest_collection_finish(session):
    """记录真实顶层用例，不预填通过数量。"""
    _items[:] = [item.nodeid for item in session.items]


def pytest_collectreport(report):
    """收集错误单独保存；缺一个测试文件不能伪装成成功。"""
    if report.failed:
        _errors.append(dict(test=report.nodeid, error=str(report.longrepr)))


def pytest_runtest_logreport(report):
    """汇总setup/call/teardown，以失败优先；子用例不覆写父用例。"""
    status = 'failed' if report.failed else 'skipped' if report.skipped else 'passed'
    if 'subtest' in type(report).__name__.lower():
        _subtests[status] += 1
        return
    old = _records.get(report.nodeid)
    rank = {'passed': 0, 'skipped': 1, 'failed': 2}
    if old is None or rank[status] > rank[old['status']]:
        item = dict(test=report.nodeid, status=status, phase=report.when)
        if status != 'passed':
            item['detail'] = str(report.longrepr)
        _records[report.nodeid] = item


def pytest_sessionfinish(session, exitstatus):
    """pytest结束后写JSON；退出码来自pytest，不按日志字符串猜成功。"""
    target = os.environ.get('UNIFIED_MOTION_PYTEST_REPORT')
    if not target:
        return
    rows = list(_records.values())
    result = dict(total=len(_items), pytest_exit_code=int(exitstatus),
                  subtests=_subtests, collection_errors=_errors, tests=rows)
    result.update({name: sum(row['status'] == name for row in rows)
                   for name in ('passed', 'failed', 'skipped')})
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
