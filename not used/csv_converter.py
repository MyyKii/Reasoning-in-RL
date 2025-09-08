import json
import csv

infile = "logs.jsonl"    
policy_csv = "policy_data.csv"
risk_csv = "risk_data.csv"

with open(infile, "r") as f_in, \
     open(policy_csv, "w", newline="") as f_pol, \
     open(risk_csv, "w", newline="") as f_risk:

    policy_writer = csv.writer(f_pol)
    risk_writer = csv.writer(f_risk)

    policy_writer.writerow(["p", "p_dot", "theta", "theta_dot", "u"])
    risk_writer.writerow(["p", "p_dot", "theta", "theta_dot", "u", "risk"])

    for line in f_in:
        sample = json.loads(line)

        s = sample["state"]   
        a = sample["action"]
        r = sample["label"]

        # Policy-Datensatz: Inputs=state, Ziel=action
        policy_writer.writerow([*s, a])

        # Risk-Datensatz: Inputs=state+action, Ziel=label
        risk_writer.writerow([*s, a, r])
