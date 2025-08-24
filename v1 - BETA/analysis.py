import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
import pandas as pd

class InnerSpeechAnalyzer:
    def __init__(self, model_path='path/to/your/model.h5'):
        try:
            self.model = keras.models.load_model(model_path)
            print(f"Model yüklendi / Model loaded: {model_path}")
        except:
            print(f"Model dosyası bulunamadı / Not found model file: {model_path}")
            self.model = None
    
    def analyze_model_architecture(self):
        if self.model is None:
            return
        
        print("Model Mimarisi / Model Architecture:")
        print("=" * 50)
        self.model.summary()

        # Model layer analizi / Model layer analysis
        total_params = self.model.count_params()
        trainable_params = sum([np.prod(v.shape) for v in self.model.trainable_variables])
        
        print(f"\nToplam Parametre Sayısı / Total Parameter: {total_params:,}")
        print(f"Eğitilebilir Parametre / Trainable Parameter: {trainable_params:,}")

        layer_types = {}
        for layer in self.model.layers:
            layer_type = type(layer).__name__
            layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
        
        print("\nLayer Türleri:")
        for layer_type, count in layer_types.items():
            print(f"  {layer_type}: {count}")
    
    def create_real_time_predictor(self):
        if self.model is None:
            print("Model yüklenmemiş! / Model not loaded!")
            return
        
        class RealTimePredictor:
            def __init__(self, model):
                self.model = model
                self.classes = ['Up', 'Down', 'Left', 'Right']
            
            def predict_thought(self, eeg_data):
                """
                EEG verisinden düşünceyi tahmin eder / Predicts thought from EEG data
                
                Args:
                    eeg_data: (n_timepoints, n_channels) şeklinde EEG verisi / EEG data in shape (n_timepoints, n_channels)
                
                Returns:
                    predicted_word: Tahmin edilen kelime / Predicted word
                    confidence: Güven skoru / Confidence score
                """
                eeg_batch = np.expand_dims(eeg_data, axis=0)  # (1, n_timepoints, n_channels)

                predictions = self.model.predict(eeg_batch, verbose=0)

                predicted_class_idx = np.argmax(predictions[0])
                confidence = predictions[0][predicted_class_idx]
                predicted_word = self.classes[predicted_class_idx]
                
                return predicted_word, confidence
            
            def get_all_predictions(self, eeg_data):
                eeg_batch = np.expand_dims(eeg_data, axis=0)
                predictions = self.model.predict(eeg_batch, verbose=0)
                
                results = {}
                for i, class_name in enumerate(self.classes):
                    results[class_name] = predictions[0][i]
                
                return results
        
        predictor = RealTimePredictor(self.model)
        print("Gerçek zamanlı tahminleyici hazır! / Real-time predictor is ready!")
        print("Kullanım: predictor.predict_thought(eeg_data) / Usage: predictor.predict_thought(eeg_data)")
        
        return predictor
    
    def simulate_brain_computer_interface(self, n_tests=20):
        """
        Beyin-Bilgisayar Arayüzü simülasyonu / Simulates Brain-Computer Interface
        """
        if self.model is None:
            print("Model yüklenmemiş! / Model not loaded!")
            return
        
        print("Beyin-Bilgisayar Arayüzü Simülasyonu / Brain-Computer Interface Simulation")
        print("=" * 40)
        
        predictor = self.create_real_time_predictor()
        classes = ['Up', 'Down', 'Left', 'Right']
        
        results = []
        
        for test in range(n_tests):
            # Rastgele gerçek sınıf seç / Randomly select true class
            true_class = np.random.choice(classes)
            true_idx = classes.index(true_class)
            
            # Bu sınıf için sentetik EEG verisi oluştur / Generate synthetic EEG data for this class
            n_channels, n_timepoints = 64, 250
            eeg_signal = np.zeros((n_timepoints, n_channels))
            
            for channel in range(n_channels):
                t = np.linspace(0, 1, n_timepoints)
                
                # Sınıfa özgü aktivasyon paternleri / Class-specific activation patterns
                base_signal = np.sin(2 * np.pi * 10 * t)
                
                if true_class == 'Up' and channel < n_channels // 4:
                    base_signal *= 2.0
                elif true_class == 'Down' and channel >= 3 * n_channels // 4:
                    base_signal *= 2.0
                elif true_class == 'Left' and channel % 2 == 0:
                    base_signal *= 1.5
                elif true_class == 'Right' and channel % 2 == 1:
                    base_signal *= 1.5
                
                # Gürültü ekle / Add noise
                noise = 0.1 * np.random.randn(n_timepoints)
                eeg_signal[:, channel] = base_signal + noise
            
            # Tahmin yap / Make prediction
            predicted_word, confidence = predictor.predict_thought(eeg_signal)
            
            results.append({
                'test_no': test + 1,
                'true_class': true_class,
                'predicted_class': predicted_word,
                'confidence': confidence,
                'correct': true_class == predicted_word
            })
            
            print(f"Test {test+1:2d}: Gerçek / Real={true_class:>5}, Tahmin / Prediction={predicted_word:>5}, "
                  f"Güven / Confidence={confidence:.3f}, {'✓' if true_class == predicted_word else '✗'}")

        df_results = pd.DataFrame(results)
        accuracy = df_results['correct'].mean()
        avg_confidence = df_results['confidence'].mean()
        
        print(f"\nSonuçlar / Results:")
        print(f"Doğruluk / Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"Ortalama Güven / Average Confidence: {avg_confidence:.3f}")

        print(f"\nSınıf Bazlı Performans / Class Based Performance:")
        for class_name in classes:
            class_results = df_results[df_results['true_class'] == class_name]
            class_accuracy = class_results['correct'].mean() if len(class_results) > 0 else 0
            print(f"  {class_name}: {class_accuracy:.3f}")
        
        return df_results
    
    def create_visualization_dashboard(self):
        """
        Model performansı için görselleştirme dashboard'u / Creates a visualization dashboard for model performance
        """
        print("Model Analiz Dashboard'u / Model Analysis Dashboard")
        print("=" * 30)

        results_df = self.simulate_brain_computer_interface(n_tests=100)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Doğruluk Dağılımı / Accuracy Distribution
        accuracy_by_class = results_df.groupby('true_class')['correct'].mean()
        axes[0, 0].bar(accuracy_by_class.index, accuracy_by_class.values)
        axes[0, 0].set_title('Sınıf Bazlı Doğruluk')
        axes[0, 0].set_ylabel('Doğruluk')
        axes[0, 0].set_ylim(0, 1)
        
        # 2. Güven Skoru Dağılımı / Confidence Score Distribution
        axes[0, 1].hist(results_df['confidence'], bins=20, alpha=0.7)
        axes[0, 1].set_title('Güven Skoru Dağılımı / Confidence Score Distribution')
        axes[0, 1].set_xlabel('Güven Skoru / Confidence Score')
        axes[0, 1].set_ylabel('Frekans / Frequency')
        
        # 3. Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(results_df['true_class'], results_df['predicted_class'], 
                             labels=['Up', 'Down', 'Left', 'Right'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Up', 'Down', 'Left', 'Right'],
                   yticklabels=['Up', 'Down', 'Left', 'Right'],
                   ax=axes[1, 0])
        axes[1, 0].set_title('Confusion Matrix')
        axes[1, 0].set_ylabel('Gerçek / True Class')
        axes[1, 0].set_xlabel('Tahmin / Predicted Class')
        
        # 4. Güven vs Doğruluk / Confidence vs Accuracy
        correct_confidences = results_df[results_df['correct'] == True]['confidence']
        wrong_confidences = results_df[results_df['correct'] == False]['confidence']
        
        axes[1, 1].hist([correct_confidences, wrong_confidences], 
                       bins=15, alpha=0.7, label=['Doğru', 'Yanlış'])
        axes[1, 1].set_title('Güven Skoru: Doğru vs Yanlış Tahminler / Confidence Score: Correct vs Incorrect Predictions')
        axes[1, 1].set_xlabel('Güven Skoru / Confidence Score')
        axes[1, 1].set_ylabel('Frekans / Frequency')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Dashboard kaydedildi / Dashboard saved: dashboard.png")
        
        return results_df

def main():
    """
    Ana analiz fonksiyonu / Main analysis function
    """
    print("bai-64 Mind Model Analizi / bai-64 Mind Model Analysis")
    print("=" * 40)

    analyzer = InnerSpeechAnalyzer('path/to/your/model.h5')
    
    if analyzer.model is not None:
        # Model mimarisini analiz et / Analyze model architecture
        analyzer.analyze_model_architecture()
        
        print("\n" + "="*50)
        
        # Gerçek zamanlı tahminleyici oluştur / Create real-time predictor
        predictor = analyzer.create_real_time_predictor()
        
        print("\n" + "="*50)
        
        # Beyin-bilgisayar arayüzü simülasyonu / Simulate brain-computer interface
        analyzer.create_visualization_dashboard()
        
        print("\nAnaliz tamamlandı! Görselleştirmeler kaydedildi. / Analysis completed! Visualizations saved.")
    
    else:
        print("Model dosyası bulunamadı. Önce modeli eğitin. / Model file not found. Train the model first.")

if __name__ == "__main__":
    main()