from audiosealWatermarking import watermark_audios
from wavmarkWatermarking import wavMark_watermark_audios
from spectral_analysis import spectral_analysis
from detection import run_detection
from detection_metrics import detection_metrics
from robustness import run_robustness

def main():
    watermark_audios()
    wavMark_watermark_audios()
    spectral_analysis()
    run_detection()
    detection_metrics()
    run_robustness()

if __name__ == "__main__":
    main()
