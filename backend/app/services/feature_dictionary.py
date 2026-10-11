"""
Centralized Feature Label Dictionary for FairCredit-AI.
Maps raw machine-learning feature names (numerical and one-hot categorical)
to human-readable, domain-accurate descriptions based on the authoritative
UCI Statlog German Credit and Taiwan Default Credit datasets.
"""

from typing import Dict, Optional

GERMAN_FIELD_LABELS: Dict[str, str] = {
    "checking_status": "Checking Account Status",
    "credit_history": "Credit Payment History",
    "purpose": "Credit Purpose",
    "savings_status": "Savings Account / Bonds",
    "employment_duration": "Employment Duration",
    "other_debtors_guarantors": "Other Debtors / Guarantors",
    "property": "Property / Assets",
    "other_installment_plans": "Other Installment Plans",
    "housing": "Housing Status",
    "job": "Job Qualification Level",
    "telephone": "Telephone Registration",
    "foreign_worker": "Foreign Worker Status",
    "duration_months": "Loan Duration (Months)",
    "credit_amount": "Credit Amount (DM)",
    "installment_rate": "Installment Rate (% Disposable Income)",
    "present_residence_since": "Residence Duration (Years)",
    "age": "Applicant Age (Years)",
    "existing_credits": "Existing Credits at Bank",
    "num_dependents": "Number of Dependents"
}

GERMAN_CODE_LABELS: Dict[str, str] = {
    # Checking status (Attribute 1)
    "A11": "< 0 DM",
    "A12": "0 to 200 DM",
    "A13": ">= 200 DM / Salary Account",
    "A14": "No Checking Account",
    # Credit history (Attribute 3)
    "A30": "No Credits / Paid Back Duly",
    "A31": "Credits at This Bank Paid Duly",
    "A32": "Existing Credits Paid Duly",
    "A33": "Delay in Paying Off in Past",
    "A34": "Critical Account / Other Credits",
    # Purpose (Attribute 4)
    "A40": "New Car",
    "A41": "Used Car",
    "A42": "Furniture / Equipment",
    "A43": "Radio / Television",
    "A44": "Domestic Appliances",
    "A45": "Repairs",
    "A46": "Education",
    "A48": "Retraining",
    "A49": "Business",
    "A410": "Other Purpose",
    # Savings status (Attribute 6)
    "A61": "< 100 DM",
    "A62": "100 to 500 DM",
    "A63": "500 to 1000 DM",
    "A64": ">= 1000 DM",
    "A65": "Unknown / No Savings Account",
    # Employment duration (Attribute 7)
    "A71": "Unemployed",
    "A72": "< 1 Year",
    "A73": "1 to 4 Years",
    "A74": "4 to 7 Years",
    "A75": ">= 7 Years",
    # Other debtors / guarantors (Attribute 10)
    "A101": "None",
    "A102": "Co-Applicant",
    "A103": "Guarantor",
    # Property (Attribute 12)
    "A121": "Real Estate",
    "A122": "Building Society / Life Insurance",
    "A123": "Car or Other Asset",
    "A124": "Unknown / No Property",
    # Other installment plans (Attribute 14)
    "A141": "Bank",
    "A142": "Stores",
    "A143": "None",
    # Housing (Attribute 15)
    "A151": "Rent",
    "A152": "Own",
    "A153": "Free Accommodation",
    # Job (Attribute 17)
    "A171": "Unemployed / Unskilled Non-Resident",
    "A172": "Unskilled Resident",
    "A173": "Skilled Employee / Official",
    "A174": "Management / Self-Employed",
    # Telephone (Attribute 19)
    "A191": "None",
    "A192": "Yes (Registered Under Customer)",
    # Foreign worker (Attribute 20)
    "A201": "Yes",
    "A202": "No"
}

TAIWAN_FIELD_LABELS: Dict[str, str] = {
    "limit_bal": "Credit Limit (NTD)",
    "age": "Applicant Age (Years)",
    "bill_amt1": "Bill Amount Sep (NTD)",
    "bill_amt2": "Bill Amount Aug (NTD)",
    "bill_amt3": "Bill Amount Jul (NTD)",
    "bill_amt4": "Bill Amount Jun (NTD)",
    "bill_amt5": "Bill Amount May (NTD)",
    "bill_amt6": "Bill Amount Apr (NTD)",
    "pay_amt1": "Payment Amount Sep (NTD)",
    "pay_amt2": "Payment Amount Aug (NTD)",
    "pay_amt3": "Payment Amount Jul (NTD)",
    "pay_amt4": "Payment Amount Jun (NTD)",
    "pay_amt5": "Payment Amount May (NTD)",
    "pay_amt6": "Payment Amount Apr (NTD)"
}

TAIWAN_PAY_MONTHS: Dict[str, str] = {
    "pay_0": "Sep",
    "pay_2": "Aug",
    "pay_3": "Jul",
    "pay_4": "Jun",
    "pay_5": "May",
    "pay_6": "Apr"
}

TAIWAN_PAY_STATUS_CODES: Dict[str, str] = {
    "-2": "No Consumption / Inactive",
    "-1": "Paid in Full",
    "0": "Revolving Credit Active",
    "1": "Payment Delay 1 Month",
    "2": "Payment Delay 2 Months",
    "3": "Payment Delay 3 Months",
    "4": "Payment Delay 4 Months",
    "5": "Payment Delay 5 Months",
    "6": "Payment Delay 6 Months",
    "7": "Payment Delay 7 Months",
    "8": "Payment Delay 8+ Months"
}

TAIWAN_EDUCATION_CODES: Dict[str, str] = {
    "1": "Graduate School",
    "2": "University",
    "3": "High School",
    "4": "Others",
    "0": "Other / Unrecorded",
    "5": "Other / Unrecorded",
    "6": "Other / Unrecorded"
}

TAIWAN_MARRIAGE_CODES: Dict[str, str] = {
    "1": "Married",
    "2": "Single",
    "3": "Others",
    "0": "Unrecorded"
}


def format_feature_label(raw_feature: Optional[str]) -> str:
    """
    Format technical feature column names into clean, readable labels.
    Preserves underlying feature semantics and appends technical code in parentheses.
    """
    if not raw_feature or not isinstance(raw_feature, str):
        return "Unavailable"

    clean_feature = raw_feature.strip()

    # Numerical features (e.g. num__duration_months, num__limit_bal)
    if clean_feature.startswith("num__"):
        col = clean_feature[5:]
        if col in GERMAN_FIELD_LABELS:
            return GERMAN_FIELD_LABELS[col]
        if col in TAIWAN_FIELD_LABELS:
            return TAIWAN_FIELD_LABELS[col]
        return col.replace("_", " ").title()

    # Categorical features (e.g. cat__checking_status_A14, cat__pay_0_2)
    if clean_feature.startswith("cat__"):
        rest = clean_feature[5:]
        parts = rest.split("_")
        code = parts[-1]
        col = "_".join(parts[:-1])

        # Taiwan payment status features (pay_0 to pay_6)
        if parts[0] == "pay" and len(parts) >= 3:
            pay_col = f"{parts[0]}_{parts[1]}"
            pay_code = "_".join(parts[2:])
            month = TAIWAN_PAY_MONTHS.get(pay_col, pay_col)
            desc = TAIWAN_PAY_STATUS_CODES.get(pay_code, f"Status {pay_code}")
            return f"Repayment {month}: {desc} ({pay_code})"

        # Taiwan education
        if col == "education":
            desc = TAIWAN_EDUCATION_CODES.get(code, f"Level {code}")
            return f"Education: {desc} ({code})"

        # Taiwan marriage
        if col == "marriage":
            desc = TAIWAN_MARRIAGE_CODES.get(code, f"Status {code}")
            return f"Marital Status: {desc} ({code})"

        # German categorical features
        if code in GERMAN_CODE_LABELS and col in GERMAN_FIELD_LABELS:
            field_name = GERMAN_FIELD_LABELS[col]
            desc = GERMAN_CODE_LABELS[code]
            return f"{field_name}: {desc} ({code})"

        field_name = col.replace("_", " ").title()
        return f"{field_name}: Category {code}"

    # Generic fallback
    return clean_feature.replace("_", " ").title()
