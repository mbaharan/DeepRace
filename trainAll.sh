DEVS=( $(seq 0 10 ) )

for dev in ${DEVS[*]}; do
    echo "-> Training Dev#$dev."
    ./train.py --test-dev $dev > ./logs/dev_$dev.log
done
