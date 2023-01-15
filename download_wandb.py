import wandb
import numpy as np

from argparse import ArgumentParser


parser = ArgumentParser(description="wandb logger")
parser.add_argument(
    "--run_id", type=str, help="which run to fetch the data from.", required=True
)

args = parser.parse_args()

api = wandb.Api()
run = api.run("cristianmanta/partial-cliques/" + args.run_id)

loss = []
log_p_x_eval = []
reverse_kl = []
log_p_hat_x_eval = []

if run.state == "finished":
    for i, row in run.history().iterrows():
        if not np.isnan(row["loss"]):
            loss.append(row["loss"])

        if not np.isnan(row["log_p_x_eval"]):
            log_p_x_eval.append(row["log_p_x_eval"])

        if not np.isnan(row["Reverse KL"]):
            reverse_kl.append(row["Reverse KL"])

        if not np.isnan(row["log_p_hat_x_eval"]):
            log_p_hat_x_eval.append(row["log_p_hat_x_eval"])
else:
    raise ValueError(f"The run with id {args.run_id} has not finished.")

loss = np.array(loss)
log_p_x_eval = np.array(log_p_x_eval)
reverse_kl = np.array(reverse_kl)
log_p_hat_x_eval = np.array(log_p_hat_x_eval)


np.save(f"loss_{args.run_id}", loss)
np.save(f"log_p_x_eval_{args.run_id}", log_p_x_eval)
np.save(f"reverse_kl_{args.run_id}", reverse_kl)
np.save(f"log_p_hat_x_eval_{args.run_id}", log_p_hat_x_eval)

metrics = run.history()
metrics.to_csv("metrics.csv")
