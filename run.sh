set -e
#echo "start: "
#nohup python -u train_att_CBAM_2.py > train_att_CBAM_2_1.log 2>&1 &
#wait

#echo "step train_att_CBAM_2.py finish"
#nohup python -u train_multitask_share5.py > train_multitask_share5.py.log 2>&1 &
#wait

#echo "step train_multitask_share5.py finish"
#for trainfile in train_MTN-CBAM_E-5.py train_CATT_E-5.py train_SATT_E-5.py; do
#  nohup python -u ${trainfile} > ./train_log/${trainfile}.log 2>&1 &
#  wait
#  echo "finish ${modelname}"
#done

#for trainfile in train_Baseline.py train_multitask_share5_w1.py; do
#  nohup python -u ${trainfile} > ./train_log/${trainfile}.log 2>&1 &
#  wait
#  echo "finish ${modelname}"
#done

#for trainfile in train_Baseline-small.py train_MTN-CBAM_E-5-small.py; do
#  nohup python -u ${trainfile} > ./train_log/${trainfile}.log 2>&1 &
#  wait
#  echo "finish ${trainfile}"
#done

#for trainfile in train_MTN5B.py train_MTN4B.py train_MTN3B.py train_MTN2B.py train_MTN1B.py; do
#  nohup python -u ${trainfile} > ./train_log/${trainfile}.log 2>&1 &
#  wait
#  echo "finish ${trainfile}"
#done
#nohup bash run.sh > ./run0502-1.log 2>&1 &

#for trainfile in train_MTN6B_cbam.py; do
#  nohup python -u ${trainfile} > ./train_log/${trainfile}.log 2>&1 &
#  wait
#  echo "finish ${trainfile}"
#done
#nohup bash run.sh > ./run0504-1.log
#nohup python -u train_MTN5B.py > ./train_log/train_MTN5B_1.py.log 2>&1 &

#for trainfile in train_MTN5B.py train_MTN4B.py train_MTN3B.py train_MTN2B.py train_MTN1B.py; do
#  nohup python -u ${trainfile} > ./train_log/${trainfile}.log 2>&1 &
#  wait
#  echo "finish ${trainfile}"
#done
#nohup bash run.sh > ./run0514.log 2>&1 &

#for trainfile in train_SCATT_E-5.py train_CATT_E-5.py train_SATT_E-5.py; do
#  echo "trainning ${trainfile}..."
#  nohup python -u ${trainfile} > ./train_log/${trainfile}_0511.log 2>&1 &
#  wait
#  echo "finish ${trainfile}"
#done

for trainfile in cbam_6_end.py cbam_1to5.py cbam_all.py; do
  echo "trainning ${trainfile}..."
  nohup python -u ${trainfile} > ./train_log/${trainfile}_0511.log 2>&1 &
  wait
  echo "finish ${trainfile}"
done
