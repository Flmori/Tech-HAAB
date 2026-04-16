#include <WiFi.h>
#include <PubSubClient.h>
#include <DHT.h>

// --- KONFIGURASI WIFI ---
const char* ssid = "ssid name";    
const char* password = "your pasword";    

// --- KONFIGURASI MQTT ---
const char* mqtt_server = "broker.emqx.io";
const int mqtt_port = 1883;
// [PENTING] GANTI BAGIAN 'NIM_LU' JADI NIM ATAU NAMA UNIK BIAR GAK BENTROK
const char* mqtt_topic = "proyek/it/semester5/sensor/galaxy_unique"; 

// --- KONFIGURASI PIN ---
#define DHTPIN 4      
#define PIRPIN 5      
#define LDRPIN 34     
#define DHTTYPE DHT11

// Batas cahaya (Sesuaikan setelah alat terpasang benar)
#define BATAS_CAHAYA 1500 

DHT dht(DHTPIN, DHTTYPE);
WiFiClient espClient;
PubSubClient client(espClient);

unsigned long lastMsg = 0;
#define MSG_INTERVAL 1000 

void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("Menghubungkan ke WiFi: ");
  Serial.println(ssid);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  int percobaan = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    percobaan++;
    if (percobaan > 20) {
      Serial.println("\n[!] WiFi Timeout. Cek Password/Hotspot.");
      percobaan = 0;
    }
  }

  Serial.println("\n[*] WiFi Terhubung!");
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("Konek ke MQTT...");
    String clientId = "ESP32-" + String(random(0xffff), HEX);
    
    if (client.connect(clientId.c_str())) {
      Serial.println("BERHASIL!");
    } else {
      Serial.print("Gagal, rc=");
      Serial.print(client.state());
      Serial.println(" coba lagi 5 detik");
      delay(5000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  
  // Set Pin Mode
  pinMode(PIRPIN, INPUT);
  // Khusus Pin 34-39 di ESP32 itu input only & gak ada internal pullup
  // Jadi WAJIB rangkaian resistor eksternal buat LDR
  pinMode(LDRPIN, INPUT); 
  
  dht.begin();
  setup_wifi();
  client.setServer(mqtt_server, mqtt_port);
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) setup_wifi();
  if (!client.connected()) reconnect();
  client.loop();

  unsigned long now = millis();
  if (now - lastMsg > MSG_INTERVAL) {
    lastMsg = now;

    // --- 1. BACA SENSOR ---
    float suhu = dht.readTemperature();
    float hum = dht.readHumidity();
    int gerak = digitalRead(PIRPIN);
    int nilaiLDR = analogRead(LDRPIN); 

    // Error Handling DHT
    if (isnan(suhu) || isnan(hum)) {
      suhu = 0; hum = 0;
      Serial.println("[!] Gagal baca DHT! Cek kabel.");
    }

    // --- 2. LOGIKA LDR ---
    // Logika: Nilai Analog Tinggi = Terang (Tergantung rangkaian resistor lu)
    // Cek serial monitor, kalau terang nilainya naik atau turun? Sesuaikan.
    int statusCahaya = (nilaiLDR > BATAS_CAHAYA) ? 1 : 0;

    // --- 3. KIRIM DATA ---
    // Format JSON
    String jsonOutput = "{";
    jsonOutput += "\"suhu\":" + String(suhu) + ",";
    jsonOutput += "\"lembab\":" + String(hum) + ",";
    jsonOutput += "\"gerak\":" + String(gerak) + ",";
    jsonOutput += "\"ldr_raw\":" + String(nilaiLDR) + ","; 
    jsonOutput += "\"cahaya\":" + String(statusCahaya);
    jsonOutput += "}";

    Serial.print("[SEND]: ");
    Serial.println(jsonOutput);
    
    char msg[200];
    jsonOutput.toCharArray(msg, 200);
    client.publish(mqtt_topic, msg);
  }
}
