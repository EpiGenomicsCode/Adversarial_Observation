
# Commandline arguments
SOURCE_INDEX="$1"
TARGET_LABEL="$2"
MODELID="$3"

# Configurable constants
ATTACK="poison_MNIST.py"
MAX_RETRIES=1
OUTPUT_BASE="MNIST_test_${MODELID}"
#MAX_RETRIES=10
#OUTPUT_BASE="MNIST-rand_test_${MODELID}"

#ARCH="basic"
#ARCH="adv"
#ARCH="MobileNet"
ARCH="RegNetX"

MODEL="mnist_${MODELID}.pt"
EPOCHS=10
PARTICLE_NUM=500
PARTICLE_GROWTH=1.5
OUTPUT_PATH="${OUTPUT_BASE}_${SOURCE_INDEX}"

echo "Running attack with:"
echo "  attack script:  $ATTACK"
echo "  model:          $MODEL"
echo "  arch:           $ARCH"
echo "  epochs:         $EPOCHS"
echo "  particleNum:    $PARTICLE_NUM"
echo "  maxRetries:     $MAX_RETRIES"
echo "  particleGrowth: $PARTICLE_GROWTH"
echo "  outputPath:     $OUTPUT_PATH"
echo "  targetLabel:    $TARGET_LABEL"
echo "  sourceIndex:    $SOURCE_INDEX"
echo

python "$ATTACK" \
  --modelPath "$MODEL" \
  --arch "$ARCH" \
  --epochs "$EPOCHS" \
  --particleNum "$PARTICLE_NUM" \
  --maxRetries "$MAX_RETRIES" \
  --particleGrowth "$PARTICLE_GROWTH" \
  --outputPath "$OUTPUT_PATH" \
  --targetLabel "$TARGET_LABEL" \
  --sourceIndex "$SOURCE_INDEX" \
  --startFromBaseline

# python "$ATTACK" \
#   --modelPath "$MODEL" \
#   --epochs "$EPOCHS" \
#   --particleNum "$PARTICLE_NUM" \
#   --maxRetries "$MAX_RETRIES" \
#   --particleGrowth "$PARTICLE_GROWTH" \
#   --outputPath "$OUTPUT_PATH" \
#   --targetLabel "$TARGET_LABEL" \
#   --sourceIndex "$SOURCE_INDEX" \
#   --variableInit
