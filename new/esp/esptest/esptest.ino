#include <ArduinoJson.h>
#include <WiFi.h>
#include <HTTPClient.h>

void setup() {
    Serial.begin(115200);
    Serial.println("Connecting to WiFi network");
    WiFi.begin("#Telia-51EE80", "y6H<3:f_c9!GTd4W");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("");
    Serial.println("WiFi connected");
}

void loop() {
    // Send a get request to 192.168.1.196:5000/faces
    Serial.println("Sending GET request");
    HTTPClient http;
    http.begin("http://192.168.1.196:5000/faces");
    Serial.println("GET request sent");
    int httpCode = http.GET();
    Serial.println(httpCode);
    if (httpCode > 0) {
        String payload = http.getString();
        // parse x and y from json payload
        Serial.println(payload);

        StaticJsonDocument<200> doc;
        DeserializationError error = deserializeJson(doc, payload);

        if (error) {
            Serial.print(F("deserializeJson() failed: "));
            Serial.println(error.f_str());
            return;
        }

        float x = doc["x"];
        float y = doc["y"];
        Serial.println(x);
        Serial.println(y);
    }

    http.end();
    delay(1000);
}
