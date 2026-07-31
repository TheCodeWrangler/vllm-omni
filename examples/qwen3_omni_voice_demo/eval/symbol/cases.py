"""Clinical notation test set. Each case: (category, raw_text, must_contain_when_spoken).
`expect=None` means "we only want to see what it does" (no assertion yet)."""
CASES = [
 # --- comparisons (the fix already covers these) ---
 ("compare", "Saturation is < 88.",                       ["less than"]),
 ("compare", "Hold if creatinine is <= 1.2.",             ["less than or equal to"]),
 ("compare", "Platelets >= 50.",                          ["greater than or equal to"]),
 ("compare", "Result != normal.",                          ["not equal to"]),
 ("compare", "Weight change +/- 2 kg.",                    ["plus or minus"]),
 ("compare", "Creatinine is ≤ 1.2 and platelets ≥ 50.",    ["less than or equal to","greater than or equal to"]),
 # --- units containing a slash ---
 ("units",   "Glucose 180 mg/dL.",                         ["milligrams per deciliter"]),
 ("units",   "Sodium 140 mEq/L.",                          ["milliequivalents per liter"]),
 ("units",   "Infuse at 125 mL/hr.",                       ["milliliters per hour"]),
 ("units",   "Oxygen at 4 L/min.",                         ["liters per minute"]),
 ("units",   "Heparin 18 units/kg/hr.",                    None),
 ("units",   "Dopamine 5 mcg/kg/min.",                     None),
 ("units",   "Creatinine clearance 60 mL/min/1.73 m2.",    None),
 # --- clinical shorthand with slashes: the "/" rule is a real hazard here ---
 ("shorthand","Patient is s/p appendectomy.",              None),
 ("shorthand","c/o chest pain.",                            None),
 ("shorthand","h/o diabetes.",                              None),
 ("shorthand","Admitted to r/o MI.",                         None),
 ("shorthand","Discharged w/o complications.",               None),
 ("shorthand","Continue w/ current regimen.",                None),
 ("shorthand","N/V/D since Tuesday.",                        None),
 ("shorthand","PT/INR pending.",                             None),
 ("shorthand","ER/PR positive.",                             None),
 ("shorthand","Vitals q/shift.",                             None),
 # --- ion / marker notation ---
 ("marker",  "Na+ 138, K+ 4.1.",                            None),
 ("marker",  "HCO3- is 24.",                                None),
 ("marker",  "CD4+ count 350.",                             None),
 ("marker",  "HER2+ breast cancer.",                        None),
 ("marker",  "Ca2+ within range.",                          None),
 # --- grading with plus ---
 ("grading", "2+ pitting edema.",                            None),
 ("grading", "3+ protein on dipstick.",                      None),
 ("grading", "Reflexes 1+ bilaterally.",                     None),
 # --- ranges / dates / times (documented as NOT template-fixable) ---
 ("range",   "Goal potassium 3.5-5.0.",                     None),
 ("range",   "Give 10-15 mg as needed.",                     None),
 ("date",    "Surgery on 3/14/2026.",                        None),
 ("time",    "Next dose at 08:30.",                           None),
 ("time",    "Ceftriaxone q8h.",                              None),
 ("time",    "Morphine q4-6h PRN.",                           None),
 # --- dosing route abbreviations ---
 ("route",   "Metoprolol 25 mg PO BID.",                     None),
 ("route",   "Ceftriaxone 1 g IV daily.",                     None),
 ("route",   "Enoxaparin 40 mg SubQ nightly.",                None),
 # --- numerals in arithmetic (known-bad, for reference) ---
 ("arith",   "100 + 10 = 110.",                               None),
 ("arith",   "100 plus 10 equals 110.",                        None),
 # --- misc symbols ---
 ("misc",    "Temp 38.5°C, EF 45%.",                          ["degrees","percent"]),
 ("misc",    "Hgb ↓ to 7.2, WBC ↑ to 14.",                    ["decreased","increased"]),
 ("misc",    "Titrate 5 → 10 mg.",                            ["to"]),
 ("misc",    "Δ mental status.",                               ["change in"]),
 ("misc",    "Dose ~ 5 mg.",                                   ["approximately"]),
 # --- drug names (pronunciation, not symbols) ---
 ("drug",    "Start hydrochlorothiazide 25 mg daily.",          None),
 ("drug",    "Levothyroxine 88 mcg every morning.",              None),
 # --- roman numerals ---
 ("roman",   "Factor VIII deficiency.",                          None),
 ("roman",   "NYHA class III heart failure.",                     None),
]
