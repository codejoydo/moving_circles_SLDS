l=$1
r=$2
for i in `seq ${l} 1 ${r}`
do
    echo $i
    python fit_rSLDS_resample.py $i 
    wait;
done
