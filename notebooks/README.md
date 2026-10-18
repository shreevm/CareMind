# MedGemma on Colab

This experiment runs MedGemma 1.5 4B on a Colab GPU and exports observations from a single chest X-ray. No local GPU is needed.

In VS Code, the notebook searches the kernel's current directory and then its parents for `.env`, reading only `HF_TOKEN`. A local kernel launched inside `E:\CareMind` can use the project's `.env`; a remote Colab kernel cannot see that local file automatically. Run `import os; print(os.getcwd())` in a cell to check the kernel directory. Model inference still requires a GPU on the kernel machine. Outside Colab, image input uses a file path and results are saved in the current directory.

1. Open https://colab.research.google.com/ and choose **File > Upload notebook**.
2. Upload `caremind_medgemma_colab.ipynb` from this folder.
3. Select **Runtime > Change runtime type > T4 GPU**, or another available GPU.
4. Accept/request access at https://huggingface.co/google/medgemma-1.5-4b-it using your Hugging Face account.
5. Create a Hugging Face read token with permission to read that gated model. Put `HF_TOKEN=your_token` in a dedicated `.env` and upload it using Colab's **Files sidebar > Upload** to `/content/.env`. Upload only this token file, not the project's full environment file. Colab Secrets and a hidden prompt are also supported. Never paste tokens into committed code or chat.
6. Run the cells in order. Start with the included public sample, then optionally upload a public/de-identified PNG or JPEG.
7. Download the JSON result using the export cell. Disconnect and delete the runtime when finished.

The notebook uses 4-bit loading to reduce memory requirements, detects the GPU's supported computation dtype, and caps output length. GPU availability is not guaranteed. A 401/403 error usually means the model access terms have not been accepted, access is pending, or token permissions are insufficient. Restart the runtime after dependency/import failures. If memory is exhausted, restart and reduce output tokens.

This is an interactive research workflow, not a live API connection to CareMind. The app's current ImagingAgent continues to use paired report/caption evidence. Generated observations are not automatically imported as verified findings. DICOM, MRI/CT volumes, PDFs, clinical validation, and live endpoint deployment are outside this notebook's scope.

Local validation checks notebook structure and Python syntax only; inference must be tested in your authenticated Colab GPU session.

Sources: [Google model card](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card), [getting started](https://developers.google.com/health-ai-developer-foundations/medgemma/get-started), [Colab limits](https://research.google.com/colaboratory/faq.html).
