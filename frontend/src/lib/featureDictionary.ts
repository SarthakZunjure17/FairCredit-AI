/**
 * Centralized Feature Label Dictionary for FairCredit-AI Frontend.
 * Maps raw machine-learning feature names (numerical and one-hot categorical)
 * to human-readable, domain-accurate descriptions based on the authoritative
 * UCI Statlog German Credit and Taiwan Default Credit datasets.
 */

export const GERMAN_FIELD_LABELS: Record<string, string> = {
  checking_status: "Checking Account Status",
  credit_history: "Credit Payment History",
  purpose: "Credit Purpose",
  savings_status: "Savings Account / Bonds",
  employment_duration: "Employment Duration",
  other_debtors_guarantors: "Other Debtors / Guarantors",
  property: "Property / Assets",
  other_installment_plans: "Other Installment Plans",
  housing: "Housing Status",
  job: "Job Qualification Level",
  telephone: "Telephone Registration",
  foreign_worker: "Foreign Worker Status",
  duration_months: "Loan Duration (Months)",
  credit_amount: "Credit Amount (DM)",
  installment_rate: "Installment Rate (% Disposable Income)",
  present_residence_since: "Residence Duration (Years)",
  age: "Applicant Age (Years)",
  existing_credits: "Existing Credits at Bank",
  num_dependents: "Number of Dependents"
};

export const GERMAN_CODE_LABELS: Record<string, string> = {
  // Checking status (Attribute 1)
  A11: "< 0 DM",
  A12: "0 to 200 DM",
  A13: ">= 200 DM / Salary Account",
  A14: "No Checking Account",
  // Credit history (Attribute 3)
  A30: "No Credits / Paid Back Duly",
  A31: "Credits at This Bank Paid Duly",
  A32: "Existing Credits Paid Duly",
  A33: "Delay in Paying Off in Past",
  A34: "Critical Account / Other Credits",
  // Purpose (Attribute 4)
  A40: "New Car",
  A41: "Used Car",
  A42: "Furniture / Equipment",
  A43: "Radio / Television",
  A44: "Domestic Appliances",
  A45: "Repairs",
  A46: "Education",
  A48: "Retraining",
  A49: "Business",
  A410: "Other Purpose",
  // Savings status (Attribute 6)
  A61: "< 100 DM",
  A62: "100 to 500 DM",
  A63: "500 to 1000 DM",
  A64: ">= 1000 DM",
  A65: "Unknown / No Savings Account",
  // Employment duration (Attribute 7)
  A71: "Unemployed",
  A72: "< 1 Year",
  A73: "1 to 4 Years",
  A74: "4 to 7 Years",
  A75: ">= 7 Years",
  // Other debtors / guarantors (Attribute 10)
  A101: "None",
  A102: "Co-Applicant",
  A103: "Guarantor",
  // Property (Attribute 12)
  A121: "Real Estate",
  A122: "Building Society / Life Insurance",
  A123: "Car or Other Asset",
  A124: "Unknown / No Property",
  // Other installment plans (Attribute 14)
  A141: "Bank",
  A142: "Stores",
  A143: "None",
  // Housing (Attribute 15)
  A151: "Rent",
  A152: "Own",
  A153: "Free Accommodation",
  // Job (Attribute 17)
  A171: "Unemployed / Unskilled Non-Resident",
  A172: "Unskilled Resident",
  A173: "Skilled Employee / Official",
  A174: "Management / Self-Employed",
  // Telephone (Attribute 19)
  A191: "None",
  A192: "Yes (Registered Under Customer)",
  // Foreign worker (Attribute 20)
  A201: "Yes",
  A202: "No"
};

export const TAIWAN_FIELD_LABELS: Record<string, string> = {
  limit_bal: "Credit Limit (NTD)",
  age: "Applicant Age (Years)",
  bill_amt1: "Bill Amount Sep (NTD)",
  bill_amt2: "Bill Amount Aug (NTD)",
  bill_amt3: "Bill Amount Jul (NTD)",
  bill_amt4: "Bill Amount Jun (NTD)",
  bill_amt5: "Bill Amount May (NTD)",
  bill_amt6: "Bill Amount Apr (NTD)",
  pay_amt1: "Payment Amount Sep (NTD)",
  pay_amt2: "Payment Amount Aug (NTD)",
  pay_amt3: "Payment Amount Jul (NTD)",
  pay_amt4: "Payment Amount Jun (NTD)",
  pay_amt5: "Payment Amount May (NTD)",
  pay_amt6: "Payment Amount Apr (NTD)"
};

export const TAIWAN_PAY_MONTHS: Record<string, string> = {
  pay_0: "Sep",
  pay_2: "Aug",
  pay_3: "Jul",
  pay_4: "Jun",
  pay_5: "May",
  pay_6: "Apr"
};

export const TAIWAN_PAY_STATUS_CODES: Record<string, string> = {
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
};

export const TAIWAN_EDUCATION_CODES: Record<string, string> = {
  "1": "Graduate School",
  "2": "University",
  "3": "High School",
  "4": "Others",
  "0": "Other / Unrecorded",
  "5": "Other / Unrecorded",
  "6": "Other / Unrecorded"
};

export const TAIWAN_MARRIAGE_CODES: Record<string, string> = {
  "1": "Married",
  "2": "Single",
  "3": "Others",
  "0": "Unrecorded"
};

/**
 * Format technical feature column names into clean, readable labels.
 * Preserves underlying feature semantics and appends technical code in parentheses.
 */
export function formatFeatureLabel(rawFeature: string | null | undefined): string {
  if (!rawFeature || typeof rawFeature !== "string") {
    return "Unavailable";
  }

  const cleanFeature = rawFeature.trim();

  // Numerical features (e.g. num__duration_months, num__limit_bal)
  if (cleanFeature.startsWith("num__")) {
    const col = cleanFeature.slice(5);
    if (GERMAN_FIELD_LABELS[col]) return GERMAN_FIELD_LABELS[col];
    if (TAIWAN_FIELD_LABELS[col]) return TAIWAN_FIELD_LABELS[col];
    return col.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  }

  // Categorical features (e.g. cat__checking_status_A14, cat__pay_0_2)
  if (cleanFeature.startsWith("cat__")) {
    const rest = cleanFeature.slice(5);
    const parts = rest.split("_");
    const code = parts[parts.length - 1];
    const col = parts.slice(0, -1).join("_");

    // Taiwan payment status features (pay_0 to pay_6)
    if (parts[0] === "pay" && parts.length >= 3) {
      const payCol = `${parts[0]}_${parts[1]}`;
      const payCode = parts.slice(2).join("_");
      const month = TAIWAN_PAY_MONTHS[payCol] || payCol;
      const desc = TAIWAN_PAY_STATUS_CODES[payCode] || `Status ${payCode}`;
      return `Repayment ${month}: ${desc} (${payCode})`;
    }

    // Taiwan education
    if (col === "education") {
      const desc = TAIWAN_EDUCATION_CODES[code] || `Level ${code}`;
      return `Education: ${desc} (${code})`;
    }

    // Taiwan marriage
    if (col === "marriage") {
      const desc = TAIWAN_MARRIAGE_CODES[code] || `Status ${code}`;
      return `Marital Status: ${desc} (${code})`;
    }

    // German categorical features
    if (GERMAN_CODE_LABELS[code] && GERMAN_FIELD_LABELS[col]) {
      const fieldName = GERMAN_FIELD_LABELS[col];
      const desc = GERMAN_CODE_LABELS[code];
      return `${fieldName}: ${desc} (${code})`;
    }

    const fieldName = col.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
    return `${fieldName}: Category ${code}`;
  }

  // Generic fallback
  return cleanFeature.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}
