
# 🇹🇷 Llama-3-8B Turkish Instruct Model (Kapalı Devre LLM)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Library](https://img.shields.io/badge/Library-Unsloth%20%26%20HuggingFace-yellow)
![Hardware](https://img.shields.io/badge/Hardware-T4%20GPU%20Compatible-green)
![License](https://img.shields.io/badge/License-MIT-red)

> **Ostim Teknik Üniversitesi - Yapay Zeka Mühendisliği**
> **Ders: Büyük Dil Modelleri (LLM)**
> **Öğrenci: Serhat Tileklioğlu (220212010)**

## 📖 Proje Özeti
Bu proje, dış kaynaklı API (OpenAI, Anthropic vb.) bağımlılığı olmadan, yerel donanım üzerinde çalışabilen Türkçe talimat takip (Instruction Following) yeteneğine sahip bir Büyük Dil Modeli geliştirmek amacıyla yapılmıştır.

Meta'nın **Llama-3-8B** modeli temel alınmış ve **Unsloth** kütüphanesi kullanılarak **QLoRA** tekniği ile optimize edilmiştir. Veri seti erişim kısıtları nedeniyle, proje kapsamında özgün bir **Sentetik Veri Üretim Hattı (Synthetic Data Pipeline)** geliştirilmiştir.

## ⚙️ Teknik Mimari ve Yöntem

Proje üç ana aşamadan oluşmaktadır:
1.  **Veri Üretimi:** Python tabanlı şablon motoru ile sentetik veri üretimi.
2.  **Fine-Tuning:** Unsloth ve LoRA ile modelin eğitilmesi.
3.  **Deployment:** Modelin GGUF formatına çevrilerek offline kullanıma hazır hale getirilmesi.

### 1. Sentetik Veri Üretimi (Engineering Solution)
Açık kaynak Türkçe veri setlerindeki erişim sorunları nedeniyle (HTTP 404/401), kural tabanlı bir veri üretim mekanizması tasarlanmıştır. Bu mekanizma ile aşağıdaki kategorilerde **2.000+** adet yüksek kaliteli eğitim verisi saniyeler içinde oluşturulmuştur:
* **Genel Kültür:** Başkent-Ülke eşleşmeleri.
* **Matematik:** Rastgele sayı üretimi ile toplama/çarpma işlemleri.
* **Teknik Sözlük:** Yazılım ve Yapay Zeka terimlerinin tanımları.

### 2. Model Optimizasyonu (QLoRA)
16GB VRAM (Tesla T4) kısıtı altında 8 milyar parametreli bir modeli eğitmek için **Quantized Low-Rank Adaptation (QLoRA)** kullanılmıştır.
* **4-bit Quantization:** Model ağırlıkları sıkıştırılarak bellek kullanımı düşürülmüştür.
* **LoRA Rank (r):** 16
* **LoRA Alpha:** 16
* **Target Modules:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

## 🚀 Kurulum ve Kullanım

Bu projeyi yerel makinenizde veya Google Colab üzerinde çalıştırmak için aşağıdaki adımları izleyin.

### Gereksinimler
```bash
pip install "unsloth[colab-new] @ git+[https://github.com/unslothai/unsloth.git](https://github.com/unslothai/unsloth.git)"
pip install --no-deps "xformers<0.0.26" trl peft accelerate bitsandbytes

```

### Modeli Çalıştırma (Python)

Eğitilmiş modeli kullanmak için örnek kod bloğu:

```python
from unsloth import FastLanguageModel

# Modeli ve Tokenizer'ı Yükle
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "model_cikti", # Veya GGUF dosya yolu
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# Inference (Soru Sorma)
FastLanguageModel.for_inference(model)
inputs = tokenizer(
    ["""### Talimat:\nDocker nedir?\n\n### Yanıt:\n"""], 
    return_tensors = "pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 128)
print(tokenizer.batch_decode(outputs)[0])

```

## 📊 Eğitim Sonuçları

Model, eğitim sürecinde şablonları başarıyla öğrenmiş ve Loss değerini dramatik şekilde düşürmüştür.

| Adım (Step) | Training Loss | Durum |
| --- | --- | --- |
| 1 | 3.1500 | Başlangıç (Rastgele Cevaplar) |
| 30 | 0.1337 | Öğrenme Aşaması |
| 60 | **0.1271** | Final (Yüksek Doğruluk) |

**Örnek Çıktılar:**

> **Soru:** Fransa'nın başkenti neresidir?
> **Model:** Fransa'nın başkenti Paris'tir.

> **Soru:** 25 ile 25 sayılarının toplamı kaçtır?
> **Model:** 25 + 25 = 50 eder.

## 📥 Model İndirme (Download)

Github dosya boyutu sınırları (Max 100MB) nedeniyle, eğitilmiş ve GGUF formatına dönüştürülmüş model dosyası harici sunucuda barındırılmaktadır.

* **Format:** GGUF (q4_k_m)
* **Boyut:** ~4.9 GB
* **Uyumluluk:** llama.cpp, LM Studio, Ollama

[👉 **MODELİ İNDİRMEK İÇİN TIKLAYIN (Google Drive)](https://colab.research.google.com/drive/1hDwyGjiReqMmWrIxJSSy9pdaQFi_H8gZ?usp=sharing)**

## 📜 Lisans

Bu proje MIT lisansı ile lisanslanmıştır. Kullanılan temel model (Llama-3) Meta'nın lisans koşullarına tabidir.

---

*Bu proje Ostim Teknik Üniversitesi Yapay Zeka Mühendisliği bölümü bitirme/ders projesi kapsamında hazırlanmıştır.*

```

