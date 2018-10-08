from Crypto.Cipher import AES
from Crypto import Random
import serial
import base64
import random
import time
import socket
import sys

##class client():
##    def __init__(self, ip_addr, port_num):
##        self.key = '3002300230023002'
##        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
##        self.block_size = 16
##        self.voltage = 0
##        self.current = 0 
##        self.power = 0 
##        self.energy = 0
##        server_address = (ip_addr, port_num)
##        print('Connecting to %s port %s' % server_address)
##        self.sock.connect(server_address)
##
##    def encrypt(self, message):
##        #print("in encrypt :   ", message) #for debugging
##        padding = ' '
##        pad = lambda s: s + (self.block_size - (len(s) % self.block_size)) * padding
##        IV = Random.new().read(AES.block_size)
##        cipher = AES.new(self.key, AES.MODE_CBC, IV)
##        paddedMessage = pad(message)
##        encoded = base64.b64encode(IV + cipher.encrypt(paddedMessage))
##        #print(encoded) #for debug
##        #print (type(encoded)) #for debug
##        return encoded
##
##    def sendData(self, message):
##        formattedAnswer = ("#"+ str(message) + "|" + str(self.voltage)\
##            + "|" + str(self.current) + "|" + str(self.power) + "|" + str(self.energy) + "|")
##        #print(formattedAnswer) #for debug
##        encryptedText = self.encrypt(formattedAnswer)
##        self.sock.send(encryptedText)
##
##if __name__ == '__main__':
##    if len(sys.argv) != 3:
##        print('Invalid number of arguments')
##        print('python server.py [IP address] [Port]')
##        sys.exit()
##
##    ip_addr = sys.argv[1]
##    port_num = int(sys.argv[2])
##    
##    pi = client(ip_addr, port_num)
####    message = ("1,5.22,4.22,6.55,2.22,999.99,20")
####    temp = []
####    powerReadings = []
####    #print(float(temp[1]))
####    for i in range (0,len(temp)):
####    temp[i] = float(temp[i])
####    #print temp
####    powerReadings = temp[1:4]
##while (1):
##    ## hardcoded data for testing of sending message
##    action = raw_input("Name of action:")
##    pi.voltage = 12
##    pi.current = 1
##    pi.power = 100
##    pi.energy = 150
##    pi.sendData(action)
##    
##    if action == 'logout':
##        print('logout, bye')
##        sys.exit(1)


def readlineCR(port):
    rv=""
    while True:
            ch=port.read().decode('utf-8')
            #print("CH is: " + ch)
            rv+= ch
            if ch=='\r' or ch=='':
                return rv
            
            


port = serial.Serial("/dev/ttyS0", baudrate = 115200, timeout=1.0)
port.reset_input_buffer()
port.reset_output_buffer()

hsflag = 0 #flag for handshaking
dataflag = 0 #flag for data transfer


numFrameDropped = 0
loopCount = 0
successCount = 0
checkSumFailCount = 0
current_milli_time = lambda: int(round(time.time() * 1000))
sentFrameID = 0
prevFrameID = 0
chksumCounter = 0
msgCheckSum = 0
computedChksum = 0
prevPred = 0
pred = 0
sendData = bytes('D', encoding = 'utf-8')
processDataFlag = 0
##serverflag = 0

while (hsflag == 0):
    print('Establishing handshake...')
    port.write("H".encode()) #send hello
    rcv = readlineCR(port)
    if(rcv == "A"):
        print("A received from Arduino MEGA")
        hsflag = 1
        print("B sent")
        port.write("B".encode())
        dataflag = 1
        
print("Handshake done!")



port.write(sendData)
port.reset_output_buffer()

while(dataflag):
        message = readlineCR(port)
        if(message):
            print("Message:", message)
            port.write(sendData)
            
        if True:
            #port.write(sendData)
            port.reset_output_buffer()
            readEndTime = current_milli_time()
            last_comma = message.rfind(",")
            #msgCheckSum = message[last_comma+1:]
            msgCheckSum = int(message.splitlines()[5].strip('\x00'))
            #print("My checksum is: ", msgCheckSum)
            sentFrameID = int(message.split(',',1)[0])
            #print("Frame id is : " ,sentFrameID)
            message = message[:last_comma+2]
            #print(message)
    
            byteMessage = bytearray(message, 'utf-8')
            
            if(sentFrameID >= (prevFrameID + 1)): #frameID is correct
                prevFrameID = sentFrameID
                while (chksumCounter < len(byteMessage)):
                    computedChksum ^= int(byteMessage[chksumCounter])
                    chksumCounter += 1
                    
                #print("Computed cs: ", computedChksum)
                if (int(computedChksum) == int(msgCheckSum)):  #checksum is correct
                    print("Matching Checksums")
                    processDataFlag = 1
##                    if (sentFrameID % 4 == 0):
##                        port.reset_input_buffer()
##                        print("input buffer reset")  #debug if input buffer is reset
                else: #checksum is wrong
                    numFrameDropped += 1
                    processDataFlag = 0; #frame is droppped
                    print("Checksum error!", "Message Checksum: ", msgCheckSum, "Generated Checksum: ", computedChksum)
                    print(' ')
          
            else: #Same or prev frameID
                numFrameDropped += 1
                processDataFlag = 0
                print('Mismatch ID!', 'Old ID: ', prevFrameID, 'New ID: ', sentFrameID)
                #print(' ')
            processDataFlag = 0
            computedChksum = 0
            chksumCounter = 0
            prevFrameID = sentFrameID
            #print("Debug Loop: ", ignoreLoopCount, "Reading has taken: ", readEndTime - readTime, "ms", "Others have taken: ", current_milli_time()-readEndTime, "ms")
            #print("Number of frames dropped: ", numFrameDropped)