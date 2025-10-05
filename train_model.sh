#!/bin/bash
set -e  # stop if any command fails

cd app

# Default short training params
EPOCHS=2
BATCH_SIZE=4
MAX_SAMPLES=10
LEARNING_RATE=0.001
MAX_STEPS=10

ARCHS=("Unet" "FPN" "PSPNet")
ENCS=("resnet18" "resnet34")
LOSSES=("DiceLoss" "BCEDice")
AUGS=("none" "single" "double" "all")

# Loop through all combinations
for arc in "${ARCHS[@]}"; do
  for enc in "${ENCS[@]}"; do
    for loss in "${LOSSES[@]}"; do
      for aug in "${AUGS[@]}"; do
        echo "🚀 Training with: $arc | $enc | $loss | $aug"
        python main.py train-model \
          -arc "$arc" \
          -enc "$enc" \
          -loss "$loss" \
          -augset "$aug" \
          -e "$EPOCHS" \
          -b "$BATCH_SIZE" \
          -s "$MAX_STEPS" \
          -m "$MAX_SAMPLES" \
          -lr "$LEARNING_RATE"
      done
    done
  done
done

echo "✅ All experiments completed."
