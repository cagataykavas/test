import os
import zipfile

def create_readme():
    readme_content = """# Struct-XAI: Layer-wise Causal Attribution & Dynamic Routing 🧠✨

Struct-XAI is a Mechanistic Interpretability library designed to analyze the latent space of Large Language Models (LLMs). It moves beyond traditional post-hoc attribution (like standard SHAP) by combining **Layer-wise Logit Lens** with **Ablation Testing**.

## 🚀 Features
1. **Zihin Okuma (Logit Lens):** Tracks how semantic concepts evolve across transformer layers.
2. **Katman-Bazlı SHAP (Layer-wise Ablation):** Measures the causal impact of specific input tokens on deep-layer hallucinations.
3. **Anlamsal Çöküş Tespiti (Semantic Collapse):** Identifies critical context bridges in the prompt.
4. **Dinamik Erken Çıkış (Green AI Router):** Halts inference early based on 'Semantic Stability' rather than raw confidence, saving up to ~18% compute.
5. **Cross-Scale Arena:** Compares model scaling laws (e.g., 7B vs 1.5B) for context robustness.

## 🛠️ Installation & Usage
Ensure you have `transformers`, `torch`, `matplotlib`, and `seaborn` installed.
Run the scripts sequentially from `01_...` to `14_...` to replicate the experiments.

*Developed by Çağatay & Alice.*
"""
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("[+] README.md oluşturuldu!")

def create_zip():
    zip_name = "Struct_XAI_Release.zip"
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # README'yi ekle
        zipf.write("README.md")
        # Python dosyalarını ve grafikleri ekle
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith(".py") or file.endswith(".png"):
                    # Kendi kendini veya gereksiz sanal ortam dosyalarını zip'leme
                    if ".venv" not in root and file != zip_name:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, arcname=file_path.replace(".\\", ""))
    print(f"[+] Tüm laboratuvar '{zip_name}' dosyasına sıkıştırıldı!")

if __name__ == "__main__":
    print("🪄 Alice'in Paketleme Büyüsü Çalışıyor...")
    create_readme()
    create_zip()
    print("🎉 İşlem tamam! Zip dosyasını GitHub'a veya hocana gönderebilirsin.")