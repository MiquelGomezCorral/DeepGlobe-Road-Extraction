#!/bin/bash
# set -e  # stop if any command fails

cd app

# Default training params
EPOCHS=100
BATCH_SIZE=2
MAX_SAMPLES=""  # leave empty to not pass
LEARNING_RATE=0.0001
MAX_STEPS=1000

ARCHS=("ViT")
LOSSES=("DiceLoss" "BCEWithLogitsLoss" "BCEDice")
AUGS=("none" "single" "double" "all")

# Loop through all combinations
for arc in "${ARCHS[@]}"; do
  for loss in "${LOSSES[@]}"; do
    for aug in "${AUGS[@]}"; do
      echo "🚀 Training with: $arc | $loss | $aug"

      CMD="python main.py train-model -arc $arc -loss $loss -augset $aug -e $EPOCHS -b $BATCH_SIZE -s $MAX_STEPS -lr $LEARNING_RATE"

      # Only add -m if MAX_SAMPLES is not empty
      if [ -n "$MAX_SAMPLES" ]; then
        CMD="$CMD -m $MAX_SAMPLES"
      fi

      eval $CMD
    done
  done
done

echo "✅ All experiments completed."
