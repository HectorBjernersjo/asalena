#define rcYPin 15
#define rcXPin 2
// #define rcPitchPin 16
#define rcRotatePin 4
#define rcShootPin 16

#define leftForwardPin 32
#define leftBackwardPin 33

#define rightForwardPin 19
#define rightBackwardPin 18

#define rotateForwardPin 22
#define rotateBackwardPin 23
#define rotateSensorPin 5


#define pitchForwardPin 26
#define pitchBackwardPin 25
#define pitchSensorPin 17

#define shootForwardPin 14
#define shootBackwardPin 12
#define shootSensorPin 13

#define turnOnPin 27

int rcYValue;
int rcXValue;

int yIsZero = 0;

int motorASpeedForward = 0;
int motorASpeedBackward = 0;

int motorBSpeedForward = 0;
int motorBSpeedBackward = 0;

const int forwardChannelA = 0;
const int backwardChannelA = 1;

const int forwardChannelB = 2;
const int backwardChannelB = 3;



int rcRotateValue;

int motorRotateSpeedForward = 0;
int motorRotateSpeedBackward = 0;

const int forwardChannelRotate = 4;
const int backwardChannelRotate = 5;



int rcPitchValue;

int motorPitchSpeedForward = 0;
int motorPitchSpeedBackward = 0;

const int forwardChannelPitch = 6;
const int backwardChannelPitch = 7;

int rcShootValue;

int motorShootSpeedForward = 0;
int motorShootSpeedBackward = 0;

const int forwardChannelShoot = 8;
const int backwardChannelShoot = 9;




int rotateSensorValue;
int pitchSensorValue;
int shootSensorValue;

const int frequency = 1000;
const int resolution = 10;

int oldRotateSensorValue = 0;

void setup() {

  Serial.begin(9600);
  delay(100);
  Serial.println("hejhoj, nu drar vi igång");

  pinMode(rcYPin, INPUT);
  pinMode(rcXPin, INPUT);
  pinMode(rcRotatePin, INPUT);
  // pinMode(rcPitchPin, INPUT);
  pinMode(rcShootPin, INPUT);

  pinMode(rotateSensorPin, INPUT);
  pinMode(pitchSensorPin, INPUT);
  pinMode(shootSensorPin, INPUT);

  ledcSetup(forwardChannelA, frequency, resolution);
  ledcSetup(backwardChannelA, frequency, resolution);

  ledcSetup(forwardChannelB, frequency, resolution);
  ledcSetup(backwardChannelB, frequency, resolution);

  ledcSetup(forwardChannelRotate, frequency, resolution);
  ledcSetup(backwardChannelRotate, frequency, resolution);

  ledcSetup(forwardChannelPitch, frequency, resolution);
  ledcSetup(backwardChannelPitch, frequency, resolution);

  ledcSetup(forwardChannelShoot, frequency, resolution);
  ledcSetup(backwardChannelShoot, frequency, resolution);

  ledcAttachPin(leftForwardPin, forwardChannelA);
  ledcAttachPin(leftBackwardPin, backwardChannelA);
  
  ledcAttachPin(rightForwardPin, forwardChannelB);
  ledcAttachPin(rightBackwardPin, backwardChannelB);

  ledcAttachPin(rotateForwardPin, forwardChannelRotate);
  ledcAttachPin(rotateBackwardPin, backwardChannelRotate);

  ledcAttachPin(pitchForwardPin, forwardChannelPitch);
  ledcAttachPin(pitchBackwardPin, backwardChannelPitch);

  ledcAttachPin(shootForwardPin, forwardChannelShoot);
  ledcAttachPin(shootBackwardPin, backwardChannelShoot);

  pinMode(turnOnPin, OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  unsigned long startTime, endTime;

  // Serial.println("Loop start");
  // rcYValue = pulseIn(rcYPin, HIGH);
  // rcXValue = pulseIn(rcXPin, HIGH);
  // rcRotateValue = pulseIn(rcRotatePin, HIGH);
  // rcPitchValue = pulseIn(rcPitchPin, HIGH);
  // rcShootValue = pulseIn(rcShootPin, HIGH);
  //
  // rotateSensorValue = pulseIn(rotateSensorPin, HIGH);
  // pitchSensorValue = pulseIn(pitchSensorPin, HIGH);
  // shootSensorValue = pulseIn(shootSensorPin, HIGH);

  // Timing rcYPin
  startTime = micros();
  rcYValue = pulseIn(rcYPin, HIGH);
  endTime = micros();
  // Serial.print("rcYPin duration: ");
  // Serial.print(endTime - startTime);
  // Serial.println(" microseconds");

  // Timing rcXPin
  startTime = micros();
  rcXValue = pulseIn(rcXPin, HIGH);
  endTime = micros();
  // Serial.print("rcXPin duration: ");
  // Serial.print(endTime - startTime);
  // Serial.println(" microseconds");

  // Timing rcRotatePin
  startTime = micros();
  rcRotateValue = pulseIn(rcRotatePin, HIGH);
  endTime = micros();
  // Serial.print("rcRotatePin duration: ");
  // Serial.print(endTime - startTime);
  // Serial.println(" microseconds");

  // Timing rcPitchPin
      // startTime = micros();
      // rcPitchValue = pulseIn(rcPitchPin, HIGH);
      // endTime = micros();
  // Serial.print("rcPitchPin duration: ");
  // Serial.print(endTime - startTime);
  // Serial.println(" microseconds");

  // Timing rcShootPin
  startTime = micros();
  rcShootValue = pulseIn(rcShootPin, HIGH);
  endTime = micros();

    Serial.print(rcYValue);
    Serial.print(",");
    Serial.print(rcXValue);
    Serial.print(",");
    Serial.print(rcRotateValue);
    Serial.print(",");
    Serial.println(rcShootValue);


  // Serial.print("rcShootPin duration: ");
  // Serial.print(endTime - startTime);
  // Serial.println(" microseconds");

  // Serial.println("Loop start 2");
  
  // // Timing rotateSensorPin
  // startTime = micros();
  // rotateSensorValue = pulseIn(rotateSensorPin, HIGH);
  // endTime = micros();
  // Serial.print("rotateSensorPin duration: ");
  // Serial.print(endTime - startTime);
  // Serial.println(" microseconds");
  //
  // // Timing pitchSensorPin
  // startTime = micros();
  // pitchSensorValue = pulseIn(pitchSensorPin, HIGH);
  // endTime = micros();
  // Serial.print("pitchSensorPin duration: ");
  // Serial.print(endTime - startTime);
  // Serial.println(" microseconds");
  //
  // // Timing shootSensorPin
  // startTime = micros();
  // shootSensorValue = pulseIn(shootSensorPin, HIGH);
  // endTime = micros();
  // Serial.print("shootSensorPin duration: ");
  // Serial.print(endTime - startTime);
  // Serial.println(" microseconds");


  digitalWrite(turnOnPin, HIGH);

  // Serial.println(rcXValue)
  // Serial.println(rcYValue);
  
    // Serial.println("Loop middle");
  
  if (rcYValue > 1510) {
    yIsZero = 1;
    motorASpeedForward = map(rcYValue, 1510, 2050, 0, 1023);
    motorASpeedBackward = 0;

    motorBSpeedForward = map(rcYValue, 1510, 2050, 0, 1023);
    motorBSpeedBackward = 0;
  }
  else if (rcYValue < 1490) { 
    yIsZero = 1;
    motorASpeedForward = 0;
    motorASpeedBackward = map(rcYValue, 950, 1490, 1023, 0);

    motorBSpeedForward = 0;
    motorBSpeedBackward = map(rcYValue, 950, 1490, 1023, 0);
  }
  else {
    yIsZero = 0;
    motorASpeedForward = 0;
    motorASpeedBackward = 0;

    motorBSpeedForward = 0;
    motorBSpeedBackward = 0;
  }

  
  if (rcXValue > 1510) {
    int turnValue = map(rcXValue, 1510, 2050, 0, 1023);
    if ( motorASpeedForward != 0 && motorBSpeedForward != 0) {
      motorASpeedForward = motorASpeedForward + turnValue;
      motorBSpeedForward = motorBSpeedForward - turnValue;
    }

    if (motorASpeedBackward != 0 && motorBSpeedBackward != 0) {
      motorASpeedBackward = motorASpeedBackward + turnValue;
      motorBSpeedBackward = motorBSpeedBackward - turnValue;
    }

    if (yIsZero == 0) {
      motorASpeedForward = turnValue;
      motorBSpeedBackward = turnValue;
    }
  }
  else if (rcXValue < 1490) {
    int turnValue = map(rcXValue, 950, 1490, 1023, 0);
    if ( motorASpeedForward != 0 && motorBSpeedForward != 0) {
      motorASpeedForward = motorASpeedForward - turnValue;
      motorBSpeedForward = motorBSpeedForward + turnValue;
    }

    if (motorASpeedBackward != 0 && motorBSpeedBackward != 0) {
      motorASpeedBackward = motorASpeedBackward - turnValue;
      motorBSpeedBackward = motorBSpeedBackward + turnValue;
    }

    if (yIsZero == 0) {
      motorASpeedBackward = turnValue;
      motorBSpeedForward = turnValue;
    }
  }
    
  if (motorASpeedForward < 0) {
    motorASpeedForward = 0;
  }

  if (motorASpeedBackward < 0) {
    motorASpeedBackward = 0;
  }

  if (motorASpeedForward > 1023) {
    motorASpeedForward = 1023;
  }

  if (motorASpeedBackward > 1023) {
    motorASpeedBackward = 1023;
  }
  

  if (motorBSpeedForward < 0) {
    motorBSpeedForward = 0;
  }

  if (motorBSpeedBackward < 0) {
    motorBSpeedBackward = 0;
  }

  if (motorBSpeedForward > 1023) {
    motorBSpeedForward = 1023;
  }

  if (motorBSpeedBackward > 1023) {
    motorBSpeedBackward = 1023;
  }

  ledcWrite(forwardChannelA, motorASpeedForward);
  ledcWrite(backwardChannelA, motorASpeedBackward);

  ledcWrite(forwardChannelB, motorBSpeedForward);
  ledcWrite(backwardChannelB, motorBSpeedBackward);





  

  rotateSensorValue = map(rotateSensorValue, 0, 4095, 0, 360);
  rotateSensorValue = rotateSensorValue / 5;

  if (oldRotateSensorValue == 0 || oldRotateSensorValue >= 360) {
    oldRotateSensorValue = rotateSensorValue;
  }
  else {
    oldRotateSensorValue += rotateSensorValue;
  }




  if (rcRotateValue > 1510) {
    motorRotateSpeedForward = map(rcRotateValue, 1510, 2050, 0, 1023);
    motorRotateSpeedBackward = 0;
  }
  else if (rcRotateValue < 1490) {
    motorRotateSpeedForward = 0;
    motorRotateSpeedBackward = map(rcRotateValue, 950, 1490, 1023, 0);
  } 
  else {
    motorRotateSpeedForward = 0;
    motorRotateSpeedBackward = 0;
  }

  ledcWrite(forwardChannelRotate, motorRotateSpeedForward);
  ledcWrite(backwardChannelRotate, motorRotateSpeedBackward);



    motorPitchSpeedForward = 0;
    motorPitchSpeedBackward = 0;

  //Serial.println(motorPitchSpeedBackward);

  ledcWrite(forwardChannelPitch, motorPitchSpeedForward);
  ledcWrite(backwardChannelPitch, motorPitchSpeedBackward);





  if (rcShootValue > 1510) {
    motorShootSpeedForward = map(rcShootValue, 1510, 2050, 0, 1023);
    motorShootSpeedForward = motorShootSpeedForward*(1);
    motorShootSpeedBackward = 0;
  }
  else if (rcShootValue < 1490) {
    motorShootSpeedForward = 0;
    motorShootSpeedBackward = map(rcShootValue, 950, 1490, 1023, 0);
    motorShootSpeedBackward = motorShootSpeedBackward*(1);
  } 
  else {
    motorShootSpeedForward = 0;
    motorShootSpeedBackward = 0;
  }

  //Serial.println(motorShootSpeedBackward);
    Serial.print("motorShootSpeedForward: ");
    Serial.println(motorShootSpeedForward);
    Serial.print("motorShootSpeedBackward: ");
    Serial.println(motorShootSpeedBackward);
  ledcWrite(forwardChannelShoot, motorShootSpeedForward);
  ledcWrite(backwardChannelShoot, motorShootSpeedBackward);
}


// void setup(){
//     Serial.begin(115200);
//     delay(1000);
//     Serial.println("Start");
// }
//
// void loop(){
//     Serial.println("Loop");
//     delay(1000);
// }
