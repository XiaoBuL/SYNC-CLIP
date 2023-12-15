start=0.1
end=1.0
step=0.1
current=$start
while (( $(bc <<< "$current <= $end") ))
do
    echo "Current value: $current"
    current=$(bc <<< "$current + $step")
done

