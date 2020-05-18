set -e
# Prepare Data:
python make_training_data.py
python make_val_and_test_data.py

# Train Modul:
for trainfile in train_MTN-CBAM.py; do
  echo "trainning ${trainfile}..."
  nohup python -u ${trainfile} > ./${trainfile}.log 2>&1 &
  wait
  echo "finish ${trainfile}"
done

# Test
for modelname in MTN-CBAM; do
  echo "${modelname} result:"
  python test_cbam.py ${modelname}
  wait
done
