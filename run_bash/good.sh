declare -a models=("new")
declare -a CWRU_sources=("CWRU_0,CWRU_2")
declare -a JNU_sources=("JNU_0,JNU_2")
declare -a CWRU_targets=("CWRU_0" "CWRU_2")
declare -a JNU_targets=("JNU_0" "JNU_2")
# Loop over each model for CWRU dataset as source and JNU as target
for model in "${models[@]}"; do
  for source in "${CWRU_sources[@]}"; do
    for target in "${JNU_targets[@]}"; do
      python train.py --model_name="$model" --source="$source" --target="$target" --train_mode multi_source --project="GOOD" --random_state 5827
      python train.py --model_name="$model" --source="$source" --target="$target" --train_mode multi_source --project="GOOD" --random_state 2463
      python train.py --model_name="$model" --source="$source" --target="$target" --train_mode multi_source --project="GOOD" --random_state 927
      python train.py --model_name="$model" --source="$source" --target="$target" --train_mode multi_source --project="GOOD" --random_state 1025
      python train.py --model_name="$model" --source="$source" --target="$target" --train_mode multi_source --project="GOOD" --random_state 2109
    done
  done
done
# Loop over each model for JNU dataset as source and CWRU as target
# for model in "${models[@]}"; do
#   for source in "${JNU_sources[@]}"; do
#     for target in "${CWRU_targets[@]}"; do
#       python train.py --model_name="$model" --source="$source" --target="$target" --train_mode source_combine --project="experiment" --random_state 5827
#       python train.py --model_name="$model" --source="$source" --target="$target" --train_mode source_combine --project="experiment" --random_state 2463
#       python train.py --model_name="$model" --source="$source" --target="$target" --train_mode source_combine --project="experiment" --random_state 927
#       python train.py --model_name="$model" --source="$source" --target="$target" --train_mode source_combine --project="experiment" --random_state 1025
#       python train.py --model_name="$model" --source="$source" --target="$target" --train_mode source_combine --project="experiment" --random_state 2109
#     done
#   done
# done
