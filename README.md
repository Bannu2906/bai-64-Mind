# bai-64 Mind | EEG-to-Text Model [BETA]🧠✍️

Classify imagined speech commands from EEG brain signals using deep learning.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/License-CC_BY_NC_SA_4.0-green)

## Overview

This project enables Brain-Computer Interface (BCI) applications by decoding imagined directional commands ("Up", "Down", "Left", "Right") from EEG brain signals. Users think about a direction without speaking, and the system predicts their intended command.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
import numpy as np
from tensorflow import keras

# Load pre-trained model
model = keras.models.load_model('path/to/your/model.h5')

# Your EEG data (1 second, 64 channels, 250 Hz sampling)
eeg_data = np.random.randn(250, 64)  # Replace with real EEG

# Make prediction
prediction = model.predict(eeg_data.reshape(1, 250, 64))
classes = ['Up', 'Down', 'Left', 'Right']
predicted_command = classes[np.argmax(prediction)]

print(f"Predicted command: {predicted_command}")
print(f"Confidence: {np.max(prediction):.3f}")
```

## Real-Time BCI Application

```python
from analysis import InnerSpeechAnalyzer

# Initialize predictor
analyzer = InnerSpeechAnalyzer('path/to/your/model.h5')
predictor = analyzer.create_real_time_predictor()

# Real-time loop
while True:
    eeg_data = capture_eeg_signal()  # Your EEG acquisition function
    command, confidence = predictor.predict_thought(eeg_data)
    
    if confidence > 0.8:
        execute_command(command)  # Your command execution
        print(f"Executing: {command}")
```

## Hardware Requirements

### EEG Device
- **Channels**: 64 Channels (10-20 system)
- **Sampling Rate**: 250+ Hz
- **Impedance**: <5kΩ
- **Bandwidth**: 0.5-100 Hz

### Recommended Devices
- OpenBCI Cyton + Daisy (16+ channels) (64 channels recommended)
- Emotiv EPOC X (14 channels) (64 channels recommended)
- g.tec g.USBamp (Professional) (64 channels recommended)

## Applications

- 🦽 **Assistive Technology**: Control for paralyzed patients
- 🎮 **Gaming**: Mind-controlled games and VR
- 🤖 **Robotics**: Brain-controlled robot navigation
- 💻 **Silent Computing**: Hands-free computer control
- 🧪 **Research**: Neuroscience and BCI studies

## Data Format

Your EEG data should be:
- **Shape**: (250, 64) per trial
- **Duration**: 1 second recording
- **Channels**: 64 EEG electrodes
- **Sampling**: 250 Hz
- **Classes**: ["Up", "Down", "Left", "Right"]

## Features

✅ **Ready-to-use** pre-trained model  
✅ **Real-time prediction** for BCI applications  
✅ **Custom training** with your own EEG data  
✅ **Multiple architectures** (CNN-LSTM, Transformer)  
✅ **EEG preprocessing** pipeline included  
✅ **Cross-platform** support (Windows, macOS, Linux)  

## Dependencies

```bash
tensorflow>=2.8.0,<3.0.0
scikit-learn>=1.0.0
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
mne>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

## Example Use Cases

### Wheelchair Control
```python
# User thinks "forward" → wheelchair moves forward
# User thinks "left" → wheelchair turns left
```

### Smart Home
```python
# User thinks "up" → lights turn on
# User thinks "down" → lights turn off
```

### Gaming
```python
# User thinks "right" → character moves right
# Mental commands for game control
```

## Support

- **Web Site**: [Neurazum](https://neurazum.com)
- **Email**: [contact@neurazum.com](mailto:contact@neurazum.com)

## Note

**This project is in the *BETA* phase. Use at your own risk. Due to the process, low accuracy rates may be observed. In addition, since the data belongs to <span style="color: #ff8d26; "><b>Neurazum</b></span>, the function structure may change in future models.**

## License

CC-BY-NC-SA 4.0 - see [LICENSE](https://creativecommons.org/licenses/by-nc-sa/4.0/) file for details.

### Acknowledgments

1. Neurazum's own data set was used. This data set is closed source.
2. Nieto, N., Peterson, V., Rufiner, H. L., Kamienkowski, J. E., & Spies, R. (2021).
"Thinking out loud, an open access EEG-based BCI dataset for inner speech recognition."
bioRxiv. https://doi.org/10.1101/2021.04.19.440473

---

*Enable mind-controlled technology with EEG! 🚀*

<span style="color: #ff8d26; "><b>Neurazum</b> AI Department</span>
