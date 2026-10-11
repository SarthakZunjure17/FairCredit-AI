import urllib.request
import sys

routes = [
    ("/", "Credit-Risk Assessment"),
    ("/assess", "Interactive Credit Risk Assessment"),
    ("/fairness", "Fairness Audit"),
    ("/research", "Empirical Research")
]

all_passed = True
for route, expected_text in routes:
    url = f"http://localhost:3000{route}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read().decode("utf-8")
            if resp.status != 200:
                print(f"[FAIL] {route} returned status {resp.status}")
                all_passed = False
                continue
            if expected_text not in content:
                print(f"[FAIL] {route} missing expected text '{expected_text}'")
                all_passed = False
                continue
            if "Cannot read properties of undefined" in content or "Unhandled Runtime Error" in content:
                print(f"[FAIL] {route} contains runtime error")
                all_passed = False
                continue
            print(f"[PASS 200 OK] {route:12s} rendered successfully with '{expected_text}'.")
    except Exception as e:
        print(f"[FAIL] {route} exception: {e}")
        all_passed = False

if not all_passed:
    sys.exit(1)
print("\nAll 4 Next.js frontend pages rendered flawlessly with 0 runtime errors.")
