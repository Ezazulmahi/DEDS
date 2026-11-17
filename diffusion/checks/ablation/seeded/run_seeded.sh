#!/bin/bash
# Seeded repeats of the full objective (reviewer 2, comment 7): two 5,000-step runs with recorded seeds, same settings
# as full_run2/full_run3, then the standard ablation evaluation and SRM detectability for each, then Path B resumes.
# Resumable (checkpoints every 1,000 steps); a STOP file in F:\DEDS_extra_checks stops it between steps.
A=/f/DEDS_extra_checks/ablation; S=/f/DEDS_extra_checks/seeded; X=/f/DEDS_extra_checks
stop() { [ -f $X/STOP ] && { echo "STOP file, exiting"; exit 1; }; }
for SEED in 2026 2027; do
  N=full_seed$SEED; OUT=$A/runs/$N
  if [ ! -f $OUT/final/unet.pt ]; then
    stop; echo "=== train $N $(date +%H:%M)"
    # no resume: a seeded run must run uninterrupted to be repeatable, so an interrupted one starts again
    [ -d $OUT ] && mv $OUT ${OUT}_interrupted_$(date +%Y%m%d%H%M)
    RES=""
    (cd $A/data_filtered && python -I -X utf8 $S/finetune_seeded.py --steps 5000 --checkpoint_every 1000 \
        --output_dir "$(cygpath -w $OUT)" --lambda_mid 0.005 --lambda_anchor 1.0 --seed $SEED $RES >> $A/runs/$N.log 2>&1) \
        || { echo "train $N FAILED"; exit 1; }
  fi
done
for SEED in 2026 2027; do
  N=full_seed$SEED; stop; echo "=== evaluate $N $(date +%H:%M)"
  (cd $A && python -I -X utf8 eval_variant.py $N "$(cygpath -w $A/runs/$N/final)" >> $A/runs/${N}_eval.log 2>&1) || { echo "eval $N FAILED"; exit 1; }
done
stop; echo "=== detectability $(date +%H:%M)"
python -I -X utf8 $X/revision/rev_abl_detect.py full_seed2026 full_seed2027 || { echo "detect FAILED"; exit 1; }
echo "seeded runs done $(date +%H:%M)"
stop; echo "=== Path B resumes $(date +%H:%M)"
cd /f/DEDS_secure_trials/pathB && python -u pathB.py all --workers 10 >> pathB.log 2>&1 || { echo "Path B FAILED"; exit 1; }
echo "all done $(date +%H:%M)"
