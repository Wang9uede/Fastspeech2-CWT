import os
import numpy as np
import torch
import pywt
import torchaudio
from sklearn.decomposition import TruncatedSVD
import os

def file_exists_in_folder(filename, folder_path):
    file_path = os.path.join(folder_path, filename)
    return os.path.exists(file_path)

class WaveletProcessor:
    def __init__(self, output_directory):
        self.output_directory = output_directory
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
            
    def load_wav_to_torch(filename):
      waveform, sample_rate = torchaudio.load(filename)
      return waveform, sample_rate

    def get_wavelet(self, filename):
        audio, sampling_rate = WaveletProcessor.load_wav_to_torch(filename)
        audio = audio.numpy()
        #print(audio.shape)
        cwtmatr, freqs = pywt.cwt(audio, np.arange(1, 129), 'morl')
        #print(cwtmatr.shape)


        cwtmatr = cwtmatr / 32768.0
        svd = TruncatedSVD(n_components=10)
        cwt_matrix_svd = svd.fit_transform(cwtmatr.squeeze())
        cwtmatr_svd = torch.FloatTensor(cwt_matrix_svd.astype(np.float32))

        return cwtmatr_svd

    def process_directory(self, directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".wav"):  # 假设音频文件的扩展名为.wav
                    filepath = os.path.join(root, file)
                    output_path = os.path.join(self.output_directory, file.replace(".wav", "_wavelet.pt"))
                    if file_exists_in_folder(output_path, filepath):
                      
                      
                      continue
                    
                      print(f"Processed and saved wavelet for {file} to {output_path}")
                    else:
                      result = self.get_wavelet(filepath)
                      torch.save(result, output_path)
                      print(f"Processed and saved wavelet for {file} to {output_path}")
if __name__ == "__main__":

    output_directory = "/content/FastSpeech2/wavelets"
    processor = WaveletProcessor(output_directory)
    # 替换为您的数据集目录
    dataset_directory = "/content/drive/MyDrive/Colab/dataset/LJSpeech-1.1/LJSpeech-1.1/wavs"
    processor.process_directory(dataset_directory)
