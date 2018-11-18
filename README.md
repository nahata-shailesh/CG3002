# CG3002
Design Project Dance Dance 08

## Required Libraries:
Apart from having Python3 installed, the following libraries would also be required in order to run our prediction script:
1. numpy (version 1.14.3 or later)
2. scipy (version 1.1.0 or later)
3. pandas (version 0.23.0 or later)
4. pycrypto (version 2.6.1 or later)
5. pyserial (version 3.4 or later)
6. scikit_learn (version 0.20.0 or later)

These libraries could simply be installed by using the `pip3` installer command. Just enter `pip3 install <Library name>` to install a specific package. For eg: In order to install numpy, type `pip3 install numpy` on your commmand line.

## Run Prediction Script:
Once installation of all the above packages is done, you are ready to run our prediction script `multimodel_prediction.py`. However, make sure you have your server running and note down the IP Address of your server and the port at which it is running. Then simply enter `python3 multimodel_prediction.py <server ip> <port no>` to run the script.
For eg: `python3 multimodel_prediction.py 192.168.43.230 5555`

That's it! Now go ahead and dance as our system predicts all your dance moves!
