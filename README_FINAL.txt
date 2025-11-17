1. Create logs folder
bash
cd /scratch/user/jangjlee7/myproject
mkdir -p logs

2. Download and copy the improved scripts
(Put the following files into the myproject folder)

3. Submit jobs
bash
sbatch submit_part1_unified_improved.sh
sbatch submit_part2_ensemble_improved.sh

4. Check logs
bash
tail -f logs/part1-<JOBID>_<TASKID>.out