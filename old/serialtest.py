import serial

# Set up the serial connection
ser = serial.Serial('COM3', 115200, dsrdtr=False, rtscts=False)  # Additional parameters to control DTR/RTS
ser.flushInput()
    
try:
    print("Sending start command to ESP32.")
    ser.write(b'S')  # Send command to start streaming messages
    
    while True:
        line = ser.readline().decode('utf-8').strip()  # Read a line from the serial port
        print(line)

except KeyboardInterrupt:
    # Exit the loop, send end command, and close the serial connection when Ctrl+C is pressed
    ser.write(b'E')  # Send command to stop streaming messages
    ser.close()
    print("\nSerial connection closed.")
