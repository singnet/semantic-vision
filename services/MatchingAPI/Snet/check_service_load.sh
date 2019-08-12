for i in `seq 1 1` 
do
    snet --print-traceback client call snet match-service getKP '{ "file@image": "Woods.jpg", "detector_name" : "Superpoint", "parameters" : "WTA_K 4"  }' -y &
    echo $i
done
