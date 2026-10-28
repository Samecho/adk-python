import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def failures(path):
  result = {}
  for case in ET.parse(path).getroot().iter('testcase'):
    for tag in ('failure', 'error'):
      problem = case.find(tag)
      if problem is not None:
        key = (case.get('classname'), case.get('name'))
        result[key] = problem.get('message', '').splitlines()[0]
  return result


log, candidate_xml, baseline_xml = map(Path, sys.argv[1:])
nodes = list(dict.fromkeys(
    line.split(' ', 1)[1].split(' - ', 1)[0].strip()
    for line in log.read_text().splitlines()
    if line.startswith(('FAILED tests/', 'ERROR tests/'))
))
assert nodes, 'No failing test IDs found; baseline failure is unverified'
result = subprocess.run([
    sys.executable, '-m', 'pytest', *nodes,
    '-q', '--tb=short', f'--junitxml={baseline_xml}',
], check=False)
candidate_failures = failures(candidate_xml)
baseline_failures = failures(baseline_xml)
assert result.returncode == 1, result.returncode
assert candidate_failures == baseline_failures, (candidate_failures, baseline_failures)
print(json.dumps({
    'baseline_failure_comparison': 'same test IDs and error messages',
    'failures': [{'test': '::'.join(key), 'error': value} for key, value in baseline_failures.items()],
}, indent=2))
