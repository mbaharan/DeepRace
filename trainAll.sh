DEVS=( $(seq 0 14 ) )

for dev in ${DEVS[*]}; do
    ./train.py --test-dev $dev > dev_$dev.log
done