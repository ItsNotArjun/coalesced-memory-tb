ssh into antpc:
ssh antpc@10.1.19.63 
password: 12345

ssh into the sharanga cluster:
ssh gargia@hpc.bits-hyderabad.ac.in


use this command to open a session with terminal access:
srun --partition=gpu_h100_4 --gres=gpu:1 --time=01:00:00 --pty bash -l
- this opens a session that allows you to set up your root dir, and code etc.



