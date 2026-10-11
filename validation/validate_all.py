"""
Master Automated Data Validation Suite for Responsible AI Credit Scoring Research.
Project: Evaluating the Impact of Fairness Mitigation on Explanation Consistency in Machine Learning-Based Credit Scoring

Runs validation across:
1. Taiwan Credit Card Default Dataset (validate_taiwan.py)
2. Statlog German Credit Dataset (validate_german.py)

Returns exit code 0 if all tests pass, exit code 1 if any failure occurs.
"""

import sys
import os

# Add validation directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from validate_taiwan import validate_taiwan
from validate_german import validate_german


def main():
    print("#" * 70)
    print("MASTER DATA VALIDATION SUITE — RESPONSIBLE AI CREDIT SCORING")
    print("#" * 70)
    
    taiwan_pass, taiwan_failures, taiwan_warnings = validate_taiwan()
    print("\n")
    german_pass, german_failures, german_warnings = validate_german()
    
    print("\n" + "#" * 70)
    print("FINAL VALIDATION SUMMARY")
    print("#" * 70)
    print(f"Taiwan Dataset Validation: {'PASS' if taiwan_pass else 'FAIL'}")
    if taiwan_failures:
        for f in taiwan_failures:
            print(f"  [ERROR] {f}")
    if taiwan_warnings:
        for w in taiwan_warnings:
            print(f"  [WARNING] {w}")
            
    print(f"\nGerman Dataset Validation: {'PASS' if german_pass else 'FAIL'}")
    if german_failures:
        for f in german_failures:
            print(f"  [ERROR] {f}")
    if german_warnings:
        for w in german_warnings:
            print(f"  [WARNING] {w}")
            
    all_pass = taiwan_pass and german_pass
    print("-" * 70)
    if all_pass:
        print("OVERALL DATA AUDIT STATUS: PASS WITH RESEARCH WARNINGS")
        print("All data integrity, target standardization, and schema rules satisfied.")
        print("Canonical datasets are verified and ready for experimental modeling pipelines.")
        print("#" * 70)
        sys.exit(0)
    else:
        print("OVERALL DATA AUDIT STATUS: FAIL")
        print("Fix highlighted failures before proceeding to modeling.")
        print("#" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
