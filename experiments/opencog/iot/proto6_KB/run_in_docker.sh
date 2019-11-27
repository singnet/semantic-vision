docker run -it -v $PWD:/run  singularitynet/opencog-dev:cli bash -c "sudo pip install websockets; cd /run;python3 main.py"
