/*
 * Edge-AI Air Quality Monitoring Station
 * Telemetry Tier (Arduino Uno)
 *
 * Reads raw analog voltages from MQ135, MQ7, and FC22 gas sensors,
 * pulls digital VOC/NOx indices from the SGP41, and reads PM1.0/PM2.5/PM10
 * from the PMS7003 laser dust sensor. Packages all readings into a single
 * comma-separated ASCII packet and pushes it to the Raspberry Pi over
 * USB serial UART at 9600 baud.
 *
 * Packet format:
 * <MQ135, MQ7, FC22, SGP41_VOC, SGP41_NOX, PM1_0, PM2_5, PM10>
 */

#include <Wire.h>
#include <SensirionI2cSgp41.h>
#include <SoftwareSerial.h>

// ---- Analog gas sensor pins ----
const int MQ135_PIN = A0;
const int MQ7_PIN   = A1;
const int FC22_PIN  = A2;

// ---- PMS7003 on hardware serial (RX/TX) ----
SoftwareSerial pmsSerial(2, 3); // RX, TX

// ---- SGP41 digital VOC/NOx sensor over I2C ----
SensirionI2cSgp41 sgp41;

// ---- MQ7 thermal cycle timing (low-temp measure / high-temp clean) ----
const unsigned long MQ7_LOW_PHASE_MS  = 60000;  // 1.5V, 60s
const unsigned long MQ7_HIGH_PHASE_MS = 90000;  // 5.0V, 90s
unsigned long mq7CycleStart = 0;
bool mq7LowPhase = true;

void setup() {
  Serial.begin(9600);
  pmsSerial.begin(9600);
  Wire.begin();
  sgp41.begin(Wire, SGP41_I2C_ADDR_59);

  mq7CycleStart = millis();
}

void loop() {
  // --- Manage MQ7 two-phase heating cycle ---
  unsigned long elapsed = millis() - mq7CycleStart;
  if (mq7LowPhase && elapsed >= MQ7_LOW_PHASE_MS) {
    mq7LowPhase = false;
    mq7CycleStart = millis();
  } else if (!mq7LowPhase && elapsed >= MQ7_HIGH_PHASE_MS) {
    mq7LowPhase = true;
    mq7CycleStart = millis();
  }

  // --- Read analog gas sensors ---
  int mq135Raw = analogRead(MQ135_PIN);
  int mq7Raw   = mq7LowPhase ? analogRead(MQ7_PIN) : -1; // only valid in low phase
  int fc22Raw  = analogRead(FC22_PIN);

  // --- Read SGP41 VOC/NOx digital indices ---
  uint16_t srawVoc = 0, srawNox = 0;
  uint16_t defaultRh = 0x8000, defaultT = 0x6666; // conditioning values
  sgp41.measureRawSignals(defaultRh, defaultT, srawVoc, srawNox);

  // --- Read PMS7003 particulate data ---
  int pm1_0 = 0, pm2_5 = 0, pm10 = 0;
  readPMS7003(pm1_0, pm2_5, pm10);

  // --- Serialize and transmit packet ---
  Serial.print(mq135Raw);    Serial.print(",");
  Serial.print(mq7Raw);      Serial.print(",");
  Serial.print(fc22Raw);     Serial.print(",");
  Serial.print(srawVoc);     Serial.print(",");
  Serial.print(srawNox);     Serial.print(",");
  Serial.print(pm1_0);       Serial.print(",");
  Serial.print(pm2_5);       Serial.print(",");
  Serial.println(pm10);

  delay(2000); // ~0.5 Hz telemetry rate
}

// Reads a single PMS7003 frame over software serial.
// Returns false (and leaves outputs at 0) if no valid frame is available.
bool readPMS7003(int &pm1_0, int &pm2_5, int &pm10) {
  if (pmsSerial.available() < 32) return false;
  if (pmsSerial.read() != 0x42) return false;
  if (pmsSerial.read() != 0x4D) return false;

  uint8_t buf[30];
  pmsSerial.readBytes(buf, 30);

  pm1_0 = (buf[8]  << 8) | buf[9];
  pm2_5 = (buf[10] << 8) | buf[11];
  pm10  = (buf[12] << 8) | buf[13];
  return true;
}
