#include "MPU6050.h"
#include <I2Cdev.h>
#include <Arduino_FreeRTOS.h>
#include <stdlib.h>
#include <Wire.h>

/* Function Headers */
void handshake();
void buildMessage();
void task1(void *p);
void initializeSensors();
void processData(int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t,int16_t);
void getPowerReadings();

/* Global variables */
int hsflag = 0;
int dataflag = 0;
char msgBuffer[3000];
MPU6050 accelgyro1;
MPU6050 accelgyro(0x69); // <-- use for AD0 high

/* Checksum */
int checksum = 0;
int app_checksum;
char checksum_m[5];

/* Constants */
const int CURRENT_PIN_A1 = A1;
const int VOLTAGE_PIN_A0 = A0;
const int RS = 0.1;
const int VOLTAGE_REF = 5;
unsigned int frameID = 1; //can start from 0 as well
char frameID_m[5];

/* Variables for reading sensors */
int16_t ax, ay, az;
int16_t gx, gy, gz;
int16_t a1x, a1y, a1z;
int16_t g1x, g1y, g1z;
int timePast = 0;
int previousTime = 0;
float currentValue; //intermediate
float voltageValue;

/* Variables for processed data */
float gForceX1, gForceY1, gForceZ1;
float rotX1, rotY1, rotZ1;

float gForceX2, gForceY2, gForceZ2; 
float rotX2, rotY2, rotZ2;

/* Variables for processing raw data */
float current, voltage, power, energy;

// Variables for building message string
char gforceX1_m[5], gforceY1_m[5], gforceZ1_m[5];
char rotX1_m[5], rotY1_m[5], rotZ1_m[5];
char gforceX2_m[5], gforceY2_m[5], gforceZ2_m[5];
char rotX2_m[5], rotY2_m[5], rotZ2_m[5];
char current_m[5], voltage_m[3], power_m[5], energy_m[5];

void setup() {
  // initialize both serial ports:
  Serial.begin(115200);
  Serial1.begin(115200);
  Wire.begin();
  initializeSensors();

  // Begin Handshake Protocol
  handshake();

  /* Create main task to be run */
  xTaskCreate(task1, "task1", 400, NULL, 2, NULL);
   
  /* Start Scheduler */
  vTaskStartScheduler(); 
}

void loop() {};

void task1(void *p) {

  int readByte = 0;
  unsigned int len = 0;
  unsigned int len2;
  TickType_t xLastWakeTime;
  const TickType_t xFrequency = 12;

  // Initialise the xLastWakeTime variable with the current time.
  xLastWakeTime = xTaskGetTickCount();
    
  // initiate data transfer
  while (dataflag) {
    if (Serial1.available()) {
      readByte = Serial1.read();
     }
    
    if (readByte == 'D') {
      readByte=0;

      xLastWakeTime = xTaskGetTickCount();
      for (int i=0; i<4 ; i++) {
      
        // Clear the data buffer
        strcpy(msgBuffer, "");
    
        // Add Frame ID to the start of the message string
        itoa(frameID, frameID_m, 10); // itoa() is used to convert a number into a string, using base10
        Serial.print("Sending frame: ");
        Serial.println(frameID);  
        strcpy(msgBuffer, frameID_m);
        strcat(msgBuffer, ",");
      
     
        //read and process power readings
        getPowerReadings();
        
        //read sensors, integration with hardware
        accelgyro.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
        accelgyro1.getMotion6(&a1x, &a1y, &a1z, &g1x, &g1y, &g1z);
        
        //process sendor data, integration with hardware
        processData(ax,ay,az, gx, gy, gz, a1x,a1y,a1z, g1x, g1y, g1z);
      
      
        // Create the message to be sent to RPi3
        buildMessage();
    
        // Append checksum at the back of the msgBuffer
        len = strlen(msgBuffer);
        for (int i=0; i<len ; i++) {
          checksum ^= msgBuffer[i];
        }
        app_checksum = (int)checksum;
        itoa(app_checksum, checksum_m, 10);
        strcat(msgBuffer, ",");
        strcat(msgBuffer, checksum_m);
    
        // Append newline character and Obtain final length of message string
        len2 = strlen(msgBuffer);
        msgBuffer[len2+1] = '\n';
        int msgLength = 0;
            
        //send the created message 
        while (msgLength < (len2 + 2)) {
          Serial1.write(msgBuffer[msgLength]);
          msgLength++;
        }
        
        // Increment frameID and Re-initialize checksum
        frameID++;
        checksum = 0;     
        vTaskDelayUntil(&xLastWakeTime, xFrequency);
      }
      Serial1.flush();
    }
  }
}

void buildMessage() {

  // Concatenate power readings to msgBuffer
  dtostrf(voltage, 1, 2, voltage_m);
  strcat(msgBuffer, voltage_m);
  strcat(msgBuffer, ",");

  dtostrf(current, 3, 2, current_m);
  strcat(msgBuffer, current_m);
  strcat(msgBuffer, ",");  

  dtostrf(power, 3, 2, power_m);
  strcat(msgBuffer, power_m);
  strcat(msgBuffer, ",");

  dtostrf(energy, 3, 2, energy_m);
  strcat(msgBuffer, energy_m); 
  strcat(msgBuffer, ",");
  
  // Concatenate processed sensor data to msgBuffer
  
  /* Accelerometer 1*/
  dtostrf(gForceX1, 3, 2, gforceX1_m);
  strcat(msgBuffer, gforceX1_m);
  strcat(msgBuffer, ",");
  
  dtostrf(gForceY1, 3, 2, gforceY1_m);
  strcat(msgBuffer, gforceY1_m);
  strcat(msgBuffer, ","); 

  dtostrf(gForceZ1, 3, 2, gforceZ1_m);
  strcat(msgBuffer, gforceZ1_m);
  strcat(msgBuffer, ",");   

  /* Gyroscope 1*/
  dtostrf(rotX1, 3, 2, rotX1_m); 
  strcat(msgBuffer, rotX1_m);
  strcat(msgBuffer, ",");

  dtostrf(rotY1, 3, 2, rotY1_m); 
  strcat(msgBuffer, rotY1_m);
  strcat(msgBuffer, ",");  

  dtostrf(rotZ1, 3, 2, rotZ1_m); 
  strcat(msgBuffer, rotZ1_m);
  strcat(msgBuffer, ",");

  /* Accelerometer 2*/
  dtostrf(gForceX2, 3, 2, gforceX2_m);
  strcat(msgBuffer, gforceX2_m);
  strcat(msgBuffer, ",");
  
  dtostrf(gForceY2, 3, 2, gforceY2_m);
  strcat(msgBuffer, gforceY2_m);
  strcat(msgBuffer, ","); 

  dtostrf(gForceZ2, 3, 2, gforceZ2_m);
  strcat(msgBuffer, gforceZ2_m);
  strcat(msgBuffer, ",");   

  /* Gyroscope 2*/
  dtostrf(rotX2, 3, 2, rotX2_m); 
  strcat(msgBuffer, rotX2_m);
  strcat(msgBuffer, ",");

  dtostrf(rotY2, 3, 2, rotY2_m); 
  strcat(msgBuffer, rotY2_m);
  strcat(msgBuffer, ",");  

  dtostrf(rotZ2, 3, 2, rotZ2_m); 
  strcat(msgBuffer, rotZ2_m);
}

void handshake() {
  while(!hsflag) {
    if (Serial1.available()) {
      char ch = Serial1.read();
      Serial.print("Received bit: ");
      Serial.println(ch);
      if (ch == 'H') {
        char ack = 'A';
        Serial1.write(ack);
      }
      if (ch == 'B') {
        hsflag=1;
        dataflag=1;
        Serial.print("Handshake Successful!\n");
      }
    }
  }
}

void initializeSensors() {

    accelgyro.initialize();
    accelgyro1.initialize();

    accelgyro.setXAccelOffset(562);
    accelgyro.setYAccelOffset(1626);
    accelgyro.setZAccelOffset(1412);
    accelgyro.setXGyroOffset(80);
    accelgyro.setYGyroOffset(35);
    accelgyro.setZGyroOffset(59);

    accelgyro1.setXAccelOffset(-449);
    accelgyro1.setYAccelOffset(-594);
    accelgyro1.setZAccelOffset(2249);
    accelgyro1.setXGyroOffset(-8);
    accelgyro1.setYGyroOffset(7);
    accelgyro1.setZGyroOffset(5);
  
}

void processData(int16_t ax,int16_t ay,int16_t az,int16_t gx,int16_t gy,int16_t gz,int16_t a1x,int16_t a1y,int16_t a1z,int16_t g1x,int16_t g1y,int16_t g1z) {
    
    gForceX1 = ax / 16384.0 * 9.81;
    gForceY1 = ay / 16384.0 * 9.81; 
    gForceZ1 = az / 16384.0 * 9.81;
    rotX1 = gx / 131.0;
    rotY1 = gy / 131.0; 
    rotZ1 = gz / 131.0;

    gForceX2 = a1x / 16384.0 * 9.81;
    gForceY2 = a1y / 16384.0 * 9.81; 
    gForceZ2 = a1z / 16384.0 * 9.81;
    rotX2 = g1x / 131.0;
    rotY2 = g1y / 131.0; 
    rotZ2 = g1z / 131.0;
}

void getPowerReadings() {

  currentValue = analogRead(CURRENT_PIN_A1);
  voltageValue = analogRead(VOLTAGE_PIN_A0);

  voltage = (voltageValue/1023) * 10;
  current = (currentValue * VOLTAGE_REF) / 1023;

  power = voltage * current;
  timePast = millis();
  energy += (power * (timePast - previousTime))/1000;
  previousTime = timePast;

  
}

