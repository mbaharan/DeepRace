DEVS=( $(seq 0 9 ) )

for dev in ${DEVS[*]}; do
    ./train.py --test-dev $dev > ./logs/dev_$dev.log
done
