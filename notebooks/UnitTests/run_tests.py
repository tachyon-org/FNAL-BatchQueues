import subprocess
from pathlib import Path
from datetime import datetime
import CodeAttemptFNAL_SeqVer as mod
ROOT = Path(__file__).parent
OUT = ROOT / "TestOutput"
OUT.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
txt_path = OUT / f"test_results_{timestamp}.txt"
xml_path = OUT / "results.xml"  # stable name for CI tools

cmd = [
    "pytest", "-v", "--maxfail=1", "--disable-warnings",
    f"--junitxml={xml_path}"
]

with open(txt_path, "w", encoding="utf-8") as f:
    subprocess.run(cmd, cwd=ROOT.parent, stdout=f, stderr=subprocess.STDOUT, check=False)

print(f"✅ Text results: {txt_path}")
print(f"📄 JUnit XML   : {xml_path}")
