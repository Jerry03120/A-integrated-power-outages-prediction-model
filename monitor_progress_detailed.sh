#!/bin/bash
# monitor_progress_detailed.sh - More comprehensive monitoring

echo "=========================================="
echo "Detailed Job Progress Monitor"
echo "Updated: $(date)"
echo "=========================================="
echo ""

# 1. Check currently running jobs
echo "1. Currently Running Jobs (should be 12):"
echo "----------------------------"
job_count=$(squeue -u jangjlee7 --noheader | wc -l)
echo "Total jobs running: ${job_count}"
echo ""
squeue -u jangjlee7 -o "%.10i %.20j %.8T %.10M %.9l %.6D %R" | head -20
echo ""

# 2. Check all expected configurations
echo "2. All 12 Configurations Status:"
echo "----------------------------"
total_configs=0
created_configs=0
started_training=0

for model_type in Unified Ensemble; do
    for algo in LightGBM XGBoost; do
        for lag in 1 12 24; do
            total_configs=$((total_configs + 1))
            outdir="outputs/${model_type}/${algo}/lag${lag}"
            
            status="❌ Not Started"
            details=""
            
            if [ -d "${outdir}" ]; then
                created_configs=$((created_configs + 1))
                status="📁 Dir Created"
                
                if [ -f "${outdir}/progress.log" ]; then
                    status="🟡 Initializing"
                fi
                
                if [ -f "${outdir}/training_output.log" ]; then
                    lines=$(wc -l < "${outdir}/training_output.log" 2>/dev/null || echo "0")
                    if [ "$lines" -gt 10 ]; then
                        started_training=$((started_training + 1))
                        status="🟢 Training"
                        details="(${lines} log lines)"
                    fi
                fi
                
                if [ -d "${outdir}/checkpoints" ]; then
                    ckpts=$(ls "${outdir}/checkpoints"/*.pkl 2>/dev/null | wc -l)
                    if [ "$ckpts" -gt 0 ]; then
                        status="🔵 Running"
                        details="(${ckpts} checkpoints)"
                    fi
                fi
                
                if [ -d "${outdir}/summary" ]; then
                    summaries=$(ls "${outdir}/summary"/*.csv 2>/dev/null | wc -l)
                    if [ "$summaries" -gt 0 ]; then
                        status="✅ Complete"
                        details="(${summaries} results)"
                    fi
                fi
            fi
            
            printf "%-8s %-10s lag%-3s : %s %s\n" "$model_type" "$algo" "$lag" "$status" "$details"
        done
    done
done

echo ""
echo "Summary: ${created_configs}/${total_configs} dirs created, ${started_training} actively training"
echo ""

# 3. Recent activity
echo "3. Recent Log Activity:"
echo "----------------------------"
echo "Part 1 logs (Unified):"
ls -lht logs/model_training_part1-17070203_*.out 2>/dev/null | head -6 | awk '{print "  "$9, "-", $5, "bytes"}'
echo ""
echo "Part 2 logs (Ensemble):"
ls -lht logs/model_training_part2-17070209_*.out 2>/dev/null | head -6 | awk '{print "  "$9, "-", $5, "bytes"}'
echo ""

# 4. Check if any jobs completed
echo "4. Completed Jobs:"
echo "----------------------------"
completed=$(sacct -u jangjlee7 -S today -o JobID,JobName,State,ExitCode | grep "model_training" | grep "COMPLETED" | wc -l)
failed=$(sacct -u jangjlee7 -S today -o JobID,JobName,State,ExitCode | grep "model_training" | grep "FAILED" | wc -l)
echo "Completed: ${completed}"
echo "Failed: ${failed}"
echo ""

# 5. Disk usage
echo "5. Disk Usage:"
echo "----------------------------"
if [ -d "outputs" ]; then
    echo "Total outputs size: $(du -sh outputs 2>/dev/null | cut -f1)"
    echo ""
    echo "Size by model type:"
    du -sh outputs/*/ 2>/dev/null | sort -h
fi
echo ""

echo "=========================================="
echo "Quick Actions:"
echo "----------------------------"
echo "  Watch this monitor (auto-refresh):"
echo "    watch -n 30 bash monitor_progress_detailed.sh"
echo ""
echo "  Check specific job:"
echo "    bash check_specific_job.sh Unified LightGBM 1"
echo ""
echo "  View live training:"
echo "    tail -f outputs/Unified/LightGBM/lag1/training_output.log"
echo "=========================================="