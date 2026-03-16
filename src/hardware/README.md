# Hardware

This folder contains all schematics, PCB designs, and CAD files for the device.

---

### Component Testing
- **HF-02-01: Microphone IC Testing**  
  Verify MEMS mic sensitivity and noise floor.  

- **HF-02-02: STM32H7 Testing**  
  Confirm development board boots, runs firmware, and handles acquisition.  

- **HF-02-03: LCD Touchscreen Testing**  
  Ensure the 7-inch display responds to touch and renders frames correctly.  

---

### Power
- **HF-03-01: Battery Pack Assembly**  
  Assemble 18650 battery pack with BMS.  

- **HF-03-02: Power Management**  
  Add regulation circuits for stable 5V/3.3V rails.  

- **HF-03-03/04: Electrical Budget Estimation**  
  Estimate current draw for STM32 and Raspberry Pi subsystems.  

- **HF-03-05: Power Testing**  
  Verify ≥2 hrs runtime and measure voltage sag.  

---

### Mechanical Housing
- **HF-04-01: CAD Design**  
  Design enclosure for mic array, Pi, STM32, battery, and display.  

---

### PCB Design & Testing
- **HF-05-01: Microphone Array Schematic**  
  Draft schematic for mic array connections and decoupling.  

- **HF-05-03/04: Embedded/Power Schematic & Layout**  
  Draw schematic and PCB for STM32, regulators, and connectors.  
