# PCB Defect Inspection Script

[cite_start]This script implements the PCB defect detection and classification pipeline described in the paper "A PCB Dataset for Defects Detection and Classification"[cite: 2, 3]. [cite_start]It uses image processing techniques (ORB feature matching, image alignment, binarization, XOR, morphology) to locate potential defects and a Convolutional Neural Network (CNN) based on DenseNet [cite: 376] to classify them.

## Requirements

The script requires Python 3 and the libraries listed in `requirements.txt`.

## Setup

1.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    ```
    Activate it:
    * Windows (PowerShell): `.\venv\Scripts\Activate.ps1` (You might need to run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first).
    * Windows (CMD): `.\venv\Scripts\activate.bat`
    * macOS/Linux: `source venv/bin/activate`

2.  **Install Libraries:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Place Files:**
    Make sure the following files are in the **same folder**:
    * `full_pipeline.py` (This script)
    * `best_model.h5` (The trained CNN model weights downloaded from Kaggle)
    * `Template image path` (The template image file)
    * `Test image path` (The test image file you want to inspect)

## How to Use

1.  **Modify Paths (Optional):**
    * Open `full_pipeline.py`.
    * Go to the `if __name__ == "__main__":` block at the bottom.
    * Change the `TEMPLATE_PATH` and `TEST_PATH` variables if your image filenames are different.

2.  **Run the Script:**
    * Open your terminal in the project folder.
    * Make sure your virtual environment is activated.
    * Run the script:
        ```bash
        python full_pipeline.py
        ```

3.  **View Results:**
    * The script will print the steps it's taking.
    * It will display a **summary plot** showing the intermediate images from the pipeline (aligned image, binary images, XOR result, cleaned mask, final result with bounding boxes).
    * If defects are found, it will display a **second plot** showing the individual 64x64 cropped images that were sent to the CNN, along with their predicted class and confidence score.

