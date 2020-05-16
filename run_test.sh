set -e
#echo "start: "
#nohup python -u train_att_CBAM_2.py > train_att_CBAM_2_1.log 2>&1 &
#wait

#echo "step train_att_CBAM_2.py finish"
#nohup python -u train_multitask_share5.py > train_multitask_share5.py.log 2>&1 &
#wait

#echo "step train_multitask_share5.py finish"
#echo "start:"
#for modelname in MTN-CBAM_E-5-small multitask_dual_22_small ; do
#  nohup python -u test_cbam.py ${modelname} > ./RESULT/${modelname}.log 2>&1 &
#  wait
#  echo "finish ${modelname}"
#done

#echo "start:"
for modelname in MTN1B MTN2B MTN3B MTN4B MTN5B MTN1B_CBAM MTN2B_CBAM MTN3B_CBAM MTN4B_CBAM MTN5B_CBAM; do
  echo "${modelname} result:"
  python test_cbam.py ${modelname}
  wait
  #echo "finish ${modelname}"
done
# nohup bash run_test.sh > ./run0504.log 2>&1 &